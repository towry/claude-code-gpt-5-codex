from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional
import hashlib

from litellm import GenericStreamingChunk, ModelResponse

from claude_code_proxy.route_model import ModelRoute
from claude_code_proxy.proxy_config import ENFORCE_ONE_TOOL_CALL_PER_RESPONSE


_THINKING_CONTENT_TYPES = {"thinking", "redacted_thinking"}
@dataclass(frozen=True)
class TransformFlags:
    preserve_thinking_blocks: bool
    strip_thinking_blocks: bool


@dataclass(frozen=True)
class ProviderRule:
    name: str
    when: Callable[[Any], bool]
    apply: Callable[[Any], None]


@dataclass
class RequestTransformContext:
    model_route: ModelRoute
    messages: list
    params: dict
    litellm_params: dict
    optional_params: dict
    flags: TransformFlags
    skip_system_prompt: bool = False


@dataclass
class ResponsesRequestTransformContext:
    model_route: ModelRoute
    items: list
    params: dict
    litellm_params: dict
    optional_params: dict
    flags: TransformFlags


@dataclass
class ResponseTransformContext:
    model_route: ModelRoute
    response: ModelResponse
    litellm_params: dict
    optional_params: dict
    flags: TransformFlags


@dataclass
class StreamingTransformContext:
    model_route: ModelRoute
    chunk: GenericStreamingChunk
    litellm_params: dict
    optional_params: dict
    flags: TransformFlags


def build_transform_flags(litellm_params: Optional[dict], optional_params: Optional[dict]) -> TransformFlags:
    additional = _get_additional_drop_params(litellm_params, optional_params)
    return TransformFlags(
        preserve_thinking_blocks="preserve_thinking_blocks" in additional,
        strip_thinking_blocks="strip_thinking_blocks" in additional,
    )


def apply_request_transforms(ctx: RequestTransformContext) -> None:
    _apply_rules(ctx, _REQUEST_RULES)


def apply_responses_request_transforms(ctx: ResponsesRequestTransformContext) -> None:
    _apply_rules(ctx, _RESPONSES_REQUEST_RULES)


def apply_response_transforms(ctx: ResponseTransformContext) -> None:
    _apply_rules(ctx, _RESPONSE_RULES)


def apply_streaming_transforms(ctx: StreamingTransformContext) -> None:
    _apply_rules(ctx, _STREAMING_RULES)


def _apply_rules(ctx: Any, rules: Iterable[ProviderRule]) -> None:
    for rule in rules:
        if rule.when(ctx):
            rule.apply(ctx)


def _get_additional_drop_params(litellm_params: Optional[dict], optional_params: Optional[dict]) -> set[str]:
    params: set[str] = set()

    for source in (litellm_params, optional_params, _extract_nested_litellm_params(optional_params)):
        if not isinstance(source, dict):
            continue
        params.update(_coerce_str_list(source.get("additional_drop_params")))

    return params


def _extract_nested_litellm_params(optional_params: Optional[dict]) -> Optional[dict]:
    if not isinstance(optional_params, dict):
        return None
    nested = optional_params.get("litellm_params")
    if isinstance(nested, dict):
        return nested
    return None


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if isinstance(item, str)]
    return []


def _is_thinking_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    part_type = part.get("type")
    if not isinstance(part_type, str):
        return False
    return part_type.lower() in _THINKING_CONTENT_TYPES


def _strip_thinking_from_content(content: Any) -> Any:
    if isinstance(content, list):
        return [part for part in content if not _is_thinking_part(part)]
    if isinstance(content, dict) and _is_thinking_part(content):
        return []
    return content


def _strip_thinking_blocks_from_message(message: Any) -> None:
    if not isinstance(message, dict):
        return
    message.pop("thinking_blocks", None)
    content = message.get("content")
    new_content = _strip_thinking_from_content(content)
    if new_content is not content:
        message["content"] = new_content


def _strip_thinking_blocks_from_messages(messages: list) -> None:
    for msg in messages:
        _strip_thinking_blocks_from_message(msg)


def _strip_thinking_blocks_from_response(response: ModelResponse) -> None:
    # NOTE: Avoid invalid signature errors when switching from Anthropic-compatible
    # upstreams to official Anthropic models by removing incompatible thinking blocks.
    for message in _iter_response_messages(response):
        _strip_thinking_blocks_from_message_obj(message)


def _strip_thinking_blocks_from_message_obj(message: Any) -> None:
    if isinstance(message, dict):
        _strip_thinking_blocks_from_message(message)
        return

    content = getattr(message, "content", None)
    new_content = _strip_thinking_from_content(content)
    if new_content is not content:
        setattr(message, "content", new_content)

    if hasattr(message, "thinking_blocks"):
        setattr(message, "thinking_blocks", None)


def _iter_response_messages(response: Any) -> Iterable[Any]:
    if isinstance(response, dict):
        choices = response.get("choices")
    else:
        choices = getattr(response, "choices", None)

    if not isinstance(choices, list):
        return

    for choice in choices:
        if isinstance(choice, dict):
            message = choice.get("message")
        else:
            message = getattr(choice, "message", None)
        if message is not None:
            yield message


def _strip_thinking_blocks_from_streaming_chunk(chunk: GenericStreamingChunk) -> None:
    if isinstance(chunk, dict):
        fields = chunk.get("provider_specific_fields")
    else:
        fields = getattr(chunk, "provider_specific_fields", None)

    if not isinstance(fields, dict):
        return

    fields.pop("thinking_blocks", None)
    delta = fields.get("delta")
    if isinstance(delta, dict):
        _strip_thinking_blocks_from_message(delta)
    content = fields.get("content")
    new_content = _strip_thinking_from_content(content)
    if new_content is not content:
        fields["content"] = new_content


def _normalize_tool_call_id_for_mistral(original_id: str) -> str:
    """
    Generate a Mistral-compatible tool call ID from the original ID.
    Mistral requires: alphanumeric only (a-z, A-Z, 0-9), exactly 9 characters.

    Uses SHA-256 hash to ensure deterministic mapping and collision resistance.
    """
    hash_digest = hashlib.sha256(original_id.encode()).hexdigest()
    return hash_digest[:9]


def _normalize_mistral_tool_calls(messages: list) -> list:
    """
    Normalize all tool call IDs in messages to be Mistral-compatible.
    Creates a mapping to ensure consistency between tool_calls and tool responses.
    """
    if not messages:
        return messages

    id_mapping = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                original_id = tc.get("id")
                if original_id and original_id not in id_mapping:
                    id_mapping[original_id] = _normalize_tool_call_id_for_mistral(original_id)

    normalized_messages = []
    for msg in messages:
        msg = dict(msg)

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
    """
    if not messages:
        return messages

    call_ids = set()
    call_id_to_msg_idx = {}
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id")
                if tc_id:
                    call_ids.add(tc_id)
                    call_id_to_msg_idx[tc_id] = idx

    responded_ids = set()
    for idx, msg in enumerate(messages):
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            tc_id = msg["tool_call_id"]
            if tc_id in call_ids:
                call_idx = call_id_to_msg_idx.get(tc_id, -1)
                if call_idx >= 0 and idx > call_idx:
                    responded_ids.add(tc_id)

    fixed_messages = []
    prev_role = None
    prev_had_tool_calls = False

    for msg in messages:
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
            valid_calls = [tc for tc in tool_calls if tc.get("id") in responded_ids]
            if valid_calls:
                msg = dict(msg)
                msg["tool_calls"] = valid_calls
                fixed_messages.append(msg)
                prev_had_tool_calls = True
            elif msg.get("content"):
                msg = dict(msg)
                msg.pop("tool_calls", None)
                fixed_messages.append(msg)
                prev_had_tool_calls = False
            else:
                prev_had_tool_calls = False
                prev_role = role
                continue

        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id in responded_ids:
                if prev_role == "assistant" and prev_had_tool_calls:
                    fixed_messages.append(msg)
                    prev_role = role
                    continue
                if prev_role == "tool":
                    fixed_messages.append(msg)
                    prev_role = role
                    continue
            print(
                "[_fix_tool_call_response_pairing] Skipping tool msg: "
                f"tc_id={tc_id}, prev_role={prev_role}, prev_had_tool_calls={prev_had_tool_calls}"
            )
            continue

        else:
            fixed_messages.append(msg)
            prev_had_tool_calls = False

        prev_role = role

    return fixed_messages


def _strip_respapi_metadata(params: dict) -> None:
    # NOTE: OpenAI Responses API upstreams reject Anthropic-specific metadata.
    params.pop("metadata", None)


def _strip_respapi_thinking_blocks(items: list, preserve_thinking_blocks: bool) -> None:
    if preserve_thinking_blocks:
        return
    for item in items:
        if isinstance(item, dict):
            item.pop("thinking_blocks", None)


def _is_connectivity_test(ctx: RequestTransformContext) -> bool:
    return (
        ctx.params.get("max_tokens") == 1
        and len(ctx.messages) == 1
        and ctx.messages[0].get("role") == "user"
        and ctx.messages[0].get("content") in ["quota", "test"]
    )


def _rewrite_connectivity_test(ctx: RequestTransformContext) -> None:
    ctx.params["max_tokens"] = 100
    ctx.messages[0]["role"] = "system"
    ctx.messages[0][
        "content"
    ] = "The intention of this request is to test connectivity. Please respond with a single word: OK"
    ctx.skip_system_prompt = True


def _inject_system_prompt_instructions(ctx: RequestTransformContext) -> None:
    system_prompt_items = []

    num_tools = len(ctx.params.get("tools") or []) + len(ctx.params.get("functions") or [])
    if ENFORCE_ONE_TOOL_CALL_PER_RESPONSE and num_tools > 1:
        system_prompt_items.append(
            "* When using tools, call AT MOST one tool per response. Never attempt multiple tool calls in a "
            "single response. The client does not support multiple tool calls in a single response. If multiple "
            "tools are needed, choose the next best single tool, return exactly one tool call, and wait for the "
            "next turn."
        )

    if ctx.model_route.use_responses_api:
        system_prompt_items.append(
            "* Until you're COMPLETELY done with your task, DO NOT EXPLAIN TO THE USER ANYTHING AT ALL, even if "
            "you need to correct your course of action (just use REASONING for that, which the user cannot see). "
            "A summary of your work at the very end is enough."
        )

    if system_prompt_items:
        ctx.messages.append(
            {
                "role": "system",
                "content": "IMPORTANT:\n" + "\n".join(system_prompt_items),
            }
        )


def _apply_mistral_fixes(ctx: RequestTransformContext) -> None:
    ctx.messages[:] = _fix_tool_call_response_pairing(ctx.messages)
    ctx.messages[:] = _normalize_mistral_tool_calls(ctx.messages)


_REQUEST_RULES: list[ProviderRule] = [
    ProviderRule(
        name="drop_context_management_non_anthropic",
        when=lambda ctx: not ctx.model_route.is_target_anthropic,
        apply=lambda ctx: ctx.params.pop("context_management", None),
    ),
    ProviderRule(
        name="strip_thinking_blocks_non_anthropic",
        when=lambda ctx: (not ctx.model_route.is_target_anthropic) and (not ctx.flags.preserve_thinking_blocks),
        apply=lambda ctx: _strip_thinking_blocks_from_messages(ctx.messages),
    ),
    ProviderRule(
        name="strip_thinking_blocks_flag",
        when=lambda ctx: ctx.flags.strip_thinking_blocks,
        apply=lambda ctx: _strip_thinking_blocks_from_messages(ctx.messages),
    ),
    ProviderRule(
        name="mistral_tool_pairing",
        when=lambda ctx: ctx.model_route.is_target_mistral,
        apply=_apply_mistral_fixes,
    ),
    ProviderRule(
        name="connectivity_test_rewrite",
        when=lambda ctx: (not ctx.model_route.is_target_anthropic) and _is_connectivity_test(ctx),
        apply=_rewrite_connectivity_test,
    ),
    ProviderRule(
        name="inject_system_prompt_instructions",
        when=lambda ctx: (
            (not ctx.model_route.is_target_anthropic)
            and (not ctx.model_route.is_target_mistral)
            and (not ctx.skip_system_prompt)
        ),
        apply=_inject_system_prompt_instructions,
    ),
]


_RESPONSES_REQUEST_RULES: list[ProviderRule] = [
    ProviderRule(
        name="drop_responses_metadata",
        when=lambda ctx: True,
        apply=lambda ctx: _strip_respapi_metadata(ctx.params),
    ),
    ProviderRule(
        name="drop_responses_thinking_blocks",
        when=lambda ctx: True,
        apply=lambda ctx: _strip_respapi_thinking_blocks(ctx.items, ctx.flags.preserve_thinking_blocks),
    ),
    ProviderRule(
        name="strip_thinking_blocks_flag",
        when=lambda ctx: ctx.flags.strip_thinking_blocks,
        apply=lambda ctx: _strip_thinking_blocks_from_messages(ctx.items),
    ),
]


_RESPONSE_RULES: list[ProviderRule] = [
    ProviderRule(
        name="strip_thinking_blocks_flag",
        when=lambda ctx: ctx.flags.strip_thinking_blocks,
        apply=lambda ctx: _strip_thinking_blocks_from_response(ctx.response),
    ),
]


_STREAMING_RULES: list[ProviderRule] = [
    ProviderRule(
        name="strip_thinking_blocks_flag",
        when=lambda ctx: ctx.flags.strip_thinking_blocks,
        apply=lambda ctx: _strip_thinking_blocks_from_streaming_chunk(ctx.chunk),
    ),
]
