from copy import deepcopy
from typing import AsyncGenerator, Callable, Generator, Optional, Union

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

from claude_code_proxy.route_model import ModelRoute
from claude_code_proxy.provider_transforms import (
    RequestTransformContext,
    ResponsesRequestTransformContext,
    ResponseTransformContext,
    StreamingTransformContext,
    apply_request_transforms,
    apply_responses_request_transforms,
    apply_response_transforms,
    apply_streaming_transforms,
    build_transform_flags,
)
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

        self.transform_flags = build_transform_flags(self.litellm_params, self.params_original)
        apply_request_transforms(
            RequestTransformContext(
                model_route=self.model_route,
                messages=self.messages_complapi,
                params=self.params_complapi,
                litellm_params=self.litellm_params,
                optional_params=self.params_original,
                flags=self.transform_flags,
            )
        )

        if self.model_route.use_responses_api:
            self.messages_respapi = convert_chat_messages_to_respapi(self.messages_complapi)
            self.params_respapi = convert_chat_params_to_respapi(self.params_complapi)
            apply_responses_request_transforms(
                ResponsesRequestTransformContext(
                    model_route=self.model_route,
                    items=self.messages_respapi,
                    params=self.params_respapi,
                    litellm_params=self.litellm_params,
                    optional_params=self.params_original,
                    flags=self.transform_flags,
                )
            )
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

            apply_response_transforms(
                ResponseTransformContext(
                    model_route=routed_request.model_route,
                    response=response_complapi,
                    litellm_params=routed_request.litellm_params,
                    optional_params=routed_request.params_original,
                    flags=routed_request.transform_flags,
                )
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

            apply_response_transforms(
                ResponseTransformContext(
                    model_route=routed_request.model_route,
                    response=response_complapi,
                    litellm_params=routed_request.litellm_params,
                    optional_params=routed_request.params_original,
                    flags=routed_request.transform_flags,
                )
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
                apply_streaming_transforms(
                    StreamingTransformContext(
                        model_route=routed_request.model_route,
                        chunk=generic_chunk,
                        litellm_params=routed_request.litellm_params,
                        optional_params=routed_request.params_original,
                        flags=routed_request.transform_flags,
                    )
                )

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
                apply_streaming_transforms(
                    StreamingTransformContext(
                        model_route=routed_request.model_route,
                        chunk=generic_chunk,
                        litellm_params=routed_request.litellm_params,
                        optional_params=routed_request.params_original,
                        flags=routed_request.transform_flags,
                    )
                )

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
