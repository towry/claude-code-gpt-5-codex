from copy import deepcopy
from typing import AsyncGenerator, Callable, Generator, Optional, Union
import hashlib

import httpx
import litellm
from litellm import (
    BaseResponsesAPIStreamingIterator,
    CustomLLM,
    CustomStreamWrapper,
    GenericStreamingChunk,
    HTTPHandler,
    ModelResponse,
    ModelResponseStream,
    AsyncHTTPHandler,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)

from claude_code_proxy.proxy_config import ENFORCE_ONE_TOOL_CALL_PER_RESPONSE
from claude_code_proxy.route_model import ModelRoute
from common.config import WRITE_TRACES_TO_FILES
from common.tracing_in_markdown import (
    write_request_trace,
    write_response_trace,
    write_streaming_chunk_trace,
)
from common.utils import (
    ProxyError,
    convert_chat_messages_to_respapi,
    convert_chat_params_to_respapi,
    convert_respapi_to_model_response,
    generate_timestamp_utc,
    to_generic_streaming_chunk,
    responses_eof_finalize_chunk,
)


def _build_auth_kwargs(
    api_base: Optional[str],
    api_key: Optional[str],
    litellm_params: Optional[dict],
) -> dict:
    if not api_base and isinstance(litellm_params, dict):
        api_base = litellm_params.get("api_base")
    if not api_key and isinstance(litellm_params, dict):
        api_key = litellm_params.get("api_key")

    auth_kwargs: dict = {}
    if api_base:
        auth_kwargs["api_base"] = api_base
    if api_key:
        auth_kwargs["api_key"] = api_key
    return auth_kwargs


def _strip_respapi_metadata(params_respapi: Optional[dict]) -> None:
    # NOTE: OpenAI Responses API upstreams reject Anthropic-specific metadata.
    # Always strip for Responses API since it's only used for non-anthropic models.
    if params_respapi:
        params_respapi.pop("metadata", None)


def _should_preserve_thinking_blocks(litellm_params: Optional[dict]) -> bool:
    """Check if thinking_blocks should be preserved for Anthropic-compatible upstreams."""
    if not litellm_params:
        return False
    additional_drop_params = litellm_params.get("additional_drop_params", [])
    return "preserve_thinking_blocks" in additional_drop_params


def _strip_respapi_thinking_blocks(messages_respapi: Optional[list], litellm_params: Optional[dict] = None) -> None:
    # NOTE: OpenAI Responses API upstreams reject Anthropic-specific thinking_blocks.
    # Skip stripping if preserve_thinking_blocks is set (for Anthropic-compatible upstreams).
    if _should_preserve_thinking_blocks(litellm_params):
        return
    if not messages_respapi:
        return
    for item in messages_respapi:
        if isinstance(item, dict):
            item.pop("thinking_blocks", None)


def _normalize_tool_call_id_for_mistral(original_id: str) -> str:
    """
    Generate a Mistral-compatible tool call ID from the original ID.
    Mistral requires: alphanumeric only (a-z, A-Z, 0-9), exactly 9 characters.

    Uses SHA-256 hash to ensure deterministic mapping and collision resistance.
    """
    # Create hash of original ID
    hash_digest = hashlib.sha256(original_id.encode()).hexdigest()
    # Take first 9 characters (all hex digits are alphanumeric)
    return hash_digest[:9]


def _normalize_mistral_tool_calls(messages: list) -> list:
    """
    Normalize all tool call IDs in messages to be Mistral-compatible.
    Creates a mapping to ensure consistency between tool_calls and tool responses.
    """
    if not messages:
        return messages

    # Build ID mapping for all tool calls first
    id_mapping = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                original_id = tc.get("id")
                if original_id and original_id not in id_mapping:
                    id_mapping[original_id] = _normalize_tool_call_id_for_mistral(original_id)

    # Apply mapping to both tool_calls and tool responses
    normalized_messages = []
    for msg in messages:
        msg = dict(msg)  # Don't mutate original

        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls = []
            for tc in msg["tool_calls"]:
                tc = dict(tc)
                original_id = tc.get("id")
                if original_id and original_id in id_mapping:
                    tc["id"] = id_mapping[original_id]
                tool_calls.append(tc)
            msg["tool_calls"] = tool_calls

        elif msg.get("role") == "tool":
            original_id = msg.get("tool_call_id")
            if original_id and original_id in id_mapping:
                msg["tool_call_id"] = id_mapping[original_id]

        normalized_messages.append(msg)

    return normalized_messages


def _fix_tool_call_response_pairing(messages: list) -> list:
    """
    Fix tool call/response pairing for providers like Mistral that require
    every tool_call to have a matching tool response.

    Mistral errors:
    - "Not the same number of function calls and responses"
    - "Unexpected role 'tool' after role 'system'"

    Strategy: Keep only complete pairs (tool_call + tool response).
    Also ensure tool messages only appear immediately after their assistant message.
    """
    if not messages:
        return messages

    # Pass 1: Collect all tool_call_ids from assistant messages with their indices
    call_ids = set()
    call_id_to_msg_idx = {}
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id")
                if tc_id:
                    call_ids.add(tc_id)
                    call_id_to_msg_idx[tc_id] = idx

    # Pass 2: Collect tool_call_ids that have BOTH a call AND a response after the call
    responded_ids = set()
    for idx, msg in enumerate(messages):
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            tc_id = msg["tool_call_id"]
            if tc_id in call_ids:
                # Ensure tool response comes after the assistant message
                call_idx = call_id_to_msg_idx.get(tc_id, -1)
                if call_idx >= 0 and idx > call_idx:
                    responded_ids.add(tc_id)

    # Pass 3: Build fixed messages - keep only complete pairs in correct order
    fixed_messages = []
    prev_role = None
    prev_had_tool_calls = False

    for idx, msg in enumerate(messages):
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
            # Keep only tool_calls that have responses
            valid_calls = [tc for tc in tool_calls if tc.get("id") in responded_ids]
            if valid_calls:
                msg = dict(msg)  # Don't mutate original
                msg["tool_calls"] = valid_calls
                fixed_messages.append(msg)
                prev_had_tool_calls = True
            elif msg.get("content"):
                # Keep the message but remove tool_calls if there's still content
                msg = dict(msg)
                msg.pop("tool_calls", None)
                fixed_messages.append(msg)
                prev_had_tool_calls = False
            else:
                # Skip the message entirely (no valid tool_calls, no content)
                prev_had_tool_calls = False
                prev_role = role
                continue

        elif role == "tool":
            # Only keep tool responses that:
            # 1. Have matching tool_calls in responded_ids
            # 2. Previous message was assistant (or another tool from same batch)
            tc_id = msg.get("tool_call_id")
            if tc_id in responded_ids:
                if prev_role == "assistant" and prev_had_tool_calls:
                    fixed_messages.append(msg)
                    prev_role = role
                    continue
                elif prev_role == "tool":
                    # Allow consecutive tool messages (batch responses)
                    fixed_messages.append(msg)
                    prev_role = role
                    continue
            # Skip this tool message - don't update prev_role so next tool can still check
            print(f"[_fix_tool_call_response_pairing] Skipping tool msg: tc_id={tc_id}, prev_role={prev_role}, prev_had_tool_calls={prev_had_tool_calls}")
            continue

        else:
            fixed_messages.append(msg)
            prev_had_tool_calls = False

        prev_role = role

    return fixed_messages


class RoutedRequest:
    def __init__(
        self,
        *,
        calling_method: str,
        model: str,
        messages_original: list,
        params_original: dict,
        stream: bool,
        litellm_params: Optional[dict] = None,
    ) -> None:
        self.timestamp = generate_timestamp_utc()
        self.calling_method = calling_method
        self.model_route = ModelRoute(model)
        self.litellm_params = litellm_params or {}

        self.messages_original = messages_original
        self.params_original = params_original

        self.messages_complapi = deepcopy(self.messages_original)
        self.params_complapi = deepcopy(self.params_original)

        self.params_complapi.update(self.model_route.extra_params)
        self.params_complapi["stream"] = stream

        if self.model_route.use_responses_api:
            # TODO What's a more reasonable way to decide when to unset
            #  temperature ?
            self.params_complapi.pop("temperature", None)

        # For Langfuse
        trace_name = f"{self.timestamp}-OUTBOUND-{self.calling_method}"
        self.params_complapi.setdefault("metadata", {})["trace_name"] = trace_name

        if not self.model_route.is_target_anthropic:
            self._adapt_complapi_for_non_anthropic_models()

        if self.model_route.use_responses_api:
            self.messages_respapi = convert_chat_messages_to_respapi(self.messages_complapi)
            self.params_respapi = convert_chat_params_to_respapi(self.params_complapi)
        else:
            self.messages_respapi = None
            self.params_respapi = None

        if WRITE_TRACES_TO_FILES:
            write_request_trace(
                timestamp=self.timestamp,
                calling_method=self.calling_method,
                messages_original=self.messages_original,
                params_original=self.params_original,
                messages_complapi=self.messages_complapi,
                params_complapi=self.params_complapi,
                messages_respapi=self.messages_respapi,
                params_respapi=self.params_respapi,
            )

    def _adapt_complapi_for_non_anthropic_models(self) -> None:
        """
        Perform necessary prompt injections to adjust certain requests to work with
        non-Anthropic models.
        """
        # Claude Code 2.x sends `context_management` on /v1/messages, but
        # OpenAI's ChatCompletions and Responses APIs do not support it
        # TODO How to reproduce the problem that the line below is fixing ?
        #  (This fix was contributed)
        self.params_complapi.pop("context_management", None)

        # Strip Anthropic-specific thinking_blocks from messages for non-Anthropic models
        # (e.g., Mistral rejects these with "Extra inputs are not permitted")
        # Skip if preserve_thinking_blocks is in additional_drop_params (Anthropic-compatible upstreams)
        if not _should_preserve_thinking_blocks(self.litellm_params):
            for msg in self.messages_complapi:
                if isinstance(msg, dict):
                    msg.pop("thinking_blocks", None)

        # Fix tool call/response pairing for providers like Mistral
        # (Mistral error: "Not the same number of function calls and responses")
        if self.model_route.target_model.startswith("mistral/"):
            self.messages_complapi = _fix_tool_call_response_pairing(self.messages_complapi)
            # Normalize tool call IDs to meet Mistral's format requirements:
            # - Alphanumeric only (a-z, A-Z, 0-9)
            # - Exactly 9 characters
            self.messages_complapi = _normalize_mistral_tool_calls(self.messages_complapi)

        if (
            self.params_complapi.get("max_tokens") == 1
            and len(self.messages_complapi) == 1
            and self.messages_complapi[0].get("role") == "user"
            and self.messages_complapi[0].get("content") in ["quota", "test"]
        ):
            # This is a "connectivity test" request by Claude Code => we need
            # to make sure non-Anthropic models don't fail because of exceeding
            # max_tokens
            self.params_complapi["max_tokens"] = 100
            self.messages_complapi[0]["role"] = "system"
            self.messages_complapi[0][
                "content"
            ] = "The intention of this request is to test connectivity. Please respond with a single word: OK"
            return

        # Skip system prompt injection for Mistral - it has strict message ordering
        # and doesn't allow system messages after tool messages
        if self.model_route.target_model.startswith("mistral/"):
            return

        system_prompt_items = []

        # Only add the instruction if at least two tools and/or functions are present in the request (in total)
        num_tools = len(self.params_complapi.get("tools") or []) + len(self.params_complapi.get("functions") or [])
        if ENFORCE_ONE_TOOL_CALL_PER_RESPONSE and num_tools > 1:
            # Add the single tool call instruction as the last message
            # TODO Get rid of this hack after the token conversion code in
            #  `common/utils.py` is reimplemented. (Seems that it's not the
            #  Claude Code CLI that doesn't support multiple tool calls in a
            #  single response, it's our token conversion code that doesn't.)
            system_prompt_items.append(
                "* When using tools, call AT MOST one tool per response. Never attempt multiple tool calls in a "
                "single response. The client does not support multiple tool calls in a single response. If multiple "
                "tools are needed, choose the next best single tool, return exactly one tool call, and wait for the "
                "next turn."
            )

        if self.model_route.use_responses_api:
            # TODO A temporary measure until the token conversion code is
            #  reimplemented. (Right now, whenever the model tries to
            #  communicate that it needs to correct its course of action, it
            #  just stops doing the task, which I suspect is a token conversion
            #  issue.)
            system_prompt_items.append(
                "* Until you're COMPLETELY done with your task, DO NOT EXPLAIN TO THE USER ANYTHING AT ALL, even if "
                "you need to correct your course of action (just use REASONING for that, which the user cannot see). "
                "A summary of your work at the very end is enough."
            )

        if system_prompt_items:
            # append the system prompt as the last message in the context
            self.messages_complapi.append(
                {
                    "role": "system",
                    "content": "IMPORTANT:\n" + "\n".join(system_prompt_items),
                }
            )


class ClaudeCodeRouter(CustomLLM):
    # pylint: disable=too-many-positional-arguments,too-many-locals

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ModelResponse:
        try:
            routed_request = RoutedRequest(
                calling_method="completion",
                model=model,
                messages_original=messages,
                params_original=optional_params,
                stream=False,
                litellm_params=litellm_params,
            )
            auth_kwargs = _build_auth_kwargs(api_base, api_key, litellm_params)

            if routed_request.model_route.use_responses_api:
                _strip_respapi_metadata(routed_request.params_respapi)
                _strip_respapi_thinking_blocks(routed_request.messages_respapi, litellm_params)
                response_respapi: ResponsesAPIResponse = litellm.responses(
                    # TODO Make sure all params are supported
                    model=routed_request.model_route.target_model,
                    input=routed_request.messages_respapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    **routed_request.params_respapi,
                    **auth_kwargs,
                )
                response_complapi: ModelResponse = convert_respapi_to_model_response(response_respapi)

            else:
                response_respapi = None
                response_complapi: ModelResponse = litellm.completion(
                    model=routed_request.model_route.target_model,
                    messages=routed_request.messages_complapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    # Drop any params that are not supported by the provider
                    drop_params=True,
                    **routed_request.params_complapi,
                    **auth_kwargs,
                )

            if WRITE_TRACES_TO_FILES:
                write_response_trace(
                    timestamp=routed_request.timestamp,
                    calling_method=routed_request.calling_method,
                    response_respapi=response_respapi,
                    response_complapi=response_complapi,
                )

            return response_complapi

        except Exception as e:
            raise ProxyError(e) from e

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ModelResponse:
        try:
            routed_request = RoutedRequest(
                calling_method="acompletion",
                model=model,
                messages_original=messages,
                params_original=optional_params,
                stream=False,
                litellm_params=litellm_params,
            )
            auth_kwargs = _build_auth_kwargs(api_base, api_key, litellm_params)

            if routed_request.model_route.use_responses_api:
                _strip_respapi_metadata(routed_request.params_respapi)
                _strip_respapi_thinking_blocks(routed_request.messages_respapi, litellm_params)
                response_respapi: ResponsesAPIResponse = await litellm.aresponses(
                    # TODO Make sure all params are supported
                    model=routed_request.model_route.target_model,
                    input=routed_request.messages_respapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    **routed_request.params_respapi,
                    **auth_kwargs,
                )
                response_complapi: ModelResponse = convert_respapi_to_model_response(response_respapi)

            else:
                response_respapi = None
                response_complapi: ModelResponse = await litellm.acompletion(
                    model=routed_request.model_route.target_model,
                    messages=routed_request.messages_complapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    # Drop any params that are not supported by the provider
                    drop_params=True,
                    **routed_request.params_complapi,
                    **auth_kwargs,
                )

            if WRITE_TRACES_TO_FILES:
                write_response_trace(
                    timestamp=routed_request.timestamp,
                    calling_method=routed_request.calling_method,
                    response_respapi=response_respapi,
                    response_complapi=response_complapi,
                )

            return response_complapi

        except Exception as e:
            raise ProxyError(e) from e

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Generator[GenericStreamingChunk, None, None]:
        try:
            routed_request = RoutedRequest(
                calling_method="streaming",
                model=model,
                messages_original=messages,
                params_original=optional_params,
                stream=True,
                litellm_params=litellm_params,
            )
            auth_kwargs = _build_auth_kwargs(api_base, api_key, litellm_params)

            if routed_request.model_route.use_responses_api:
                _strip_respapi_metadata(routed_request.params_respapi)
                _strip_respapi_thinking_blocks(routed_request.messages_respapi, litellm_params)
                resp_stream: BaseResponsesAPIStreamingIterator = litellm.responses(
                    # TODO Make sure all params are supported
                    model=routed_request.model_route.target_model,
                    input=routed_request.messages_respapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    **routed_request.params_respapi,
                    **auth_kwargs,
                )

            else:
                resp_stream: CustomStreamWrapper = litellm.completion(
                    model=routed_request.model_route.target_model,
                    messages=routed_request.messages_complapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    # Drop any params that are not supported by the provider
                    drop_params=True,
                    **routed_request.params_complapi,
                    **auth_kwargs,
                )

            for chunk_idx, chunk in enumerate[ModelResponseStream | ResponsesAPIStreamingResponse](resp_stream):
                generic_chunk = to_generic_streaming_chunk(chunk)

                if WRITE_TRACES_TO_FILES:
                    if routed_request.model_route.use_responses_api:
                        respapi_chunk, complapi_chunk = chunk, None
                    else:
                        respapi_chunk, complapi_chunk = None, chunk

                    write_streaming_chunk_trace(
                        timestamp=routed_request.timestamp,
                        calling_method=routed_request.calling_method,
                        chunk_idx=chunk_idx,
                        respapi_chunk=respapi_chunk,
                        complapi_chunk=complapi_chunk,
                        generic_chunk=generic_chunk,
                    )

                yield generic_chunk

            # EOF fallback: if provider ended stream without a terminal event and
            # we have a pending tool with buffered args, emit once.
            # TODO Refactor or get rid of the try/except block below after the
            #  code in `common/utils.py` is owned (after the vibe-code there is
            #  replaced with proper code)
            try:
                eof_chunk = responses_eof_finalize_chunk()
                if eof_chunk is not None:
                    yield eof_chunk
            except Exception:  # pylint: disable=broad-exception-caught
                # Ignore; best-effort fallback
                pass

        except Exception as e:
            raise ProxyError(e) from e

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AsyncGenerator[GenericStreamingChunk, None]:
        try:
            routed_request = RoutedRequest(
                calling_method="astreaming",
                model=model,
                messages_original=messages,
                params_original=optional_params,
                stream=True,
                litellm_params=litellm_params,
            )
            auth_kwargs = _build_auth_kwargs(api_base, api_key, litellm_params)

            if routed_request.model_route.use_responses_api:
                _strip_respapi_metadata(routed_request.params_respapi)
                _strip_respapi_thinking_blocks(routed_request.messages_respapi, litellm_params)
                resp_stream: BaseResponsesAPIStreamingIterator = await litellm.aresponses(
                    # TODO Make sure all params are supported
                    model=routed_request.model_route.target_model,
                    input=routed_request.messages_respapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    **routed_request.params_respapi,
                    **auth_kwargs,
                )

            else:
                resp_stream: CustomStreamWrapper = await litellm.acompletion(
                    model=routed_request.model_route.target_model,
                    messages=routed_request.messages_complapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    # Drop any params that are not supported by the provider
                    drop_params=True,
                    **routed_request.params_complapi,
                    **auth_kwargs,
                )

            chunk_idx = 0
            async for chunk in resp_stream:
                generic_chunk = to_generic_streaming_chunk(chunk)

                if WRITE_TRACES_TO_FILES:
                    if routed_request.model_route.use_responses_api:
                        respapi_chunk, complapi_chunk = chunk, None
                    else:
                        respapi_chunk, complapi_chunk = None, chunk

                    write_streaming_chunk_trace(
                        timestamp=routed_request.timestamp,
                        calling_method=routed_request.calling_method,
                        chunk_idx=chunk_idx,
                        respapi_chunk=respapi_chunk,
                        complapi_chunk=complapi_chunk,
                        generic_chunk=generic_chunk,
                    )

                yield generic_chunk
                chunk_idx += 1

            # EOF fallback: if provider ended stream without a terminal event and
            # we have a pending tool with buffered args, emit once.
            # TODO Refactor or get rid of the try/except block below after the
            #  code in `common/utils.py` is owned (after the vibe-code there is
            #  replaced with proper code)
            try:
                eof_chunk = responses_eof_finalize_chunk()
                if eof_chunk is not None:
                    yield eof_chunk
            except Exception:  # pylint: disable=broad-exception-caught
                # Ignore; best-effort fallback
                pass

        except Exception as e:
            raise ProxyError(e) from e


claude_code_router = ClaudeCodeRouter()
