# Adding New Provider Support to claude_code_router

This guide documents the process of integrating new LLM providers with the `claude_code_router` custom handler, based on the Mistral integration experience.

## Overview

The `claude_code_router` is a custom LiteLLM handler that:
1. Receives Claude Code requests (Anthropic Messages API format)
2. Transforms them for target providers
3. Routes to either Responses API or Completion API based on provider capabilities

## Key Files

| File | Purpose |
|------|---------|
| `claude_code_proxy/route_model.py` | Model routing logic, API mode selection |
| `claude_code_proxy/claude_code_router.py` | Request transformation, API calls |
| `claude_code_proxy/proxy_config.py` | Configuration constants |

## Provider Integration Checklist

### 1. API Mode Selection (`route_model.py`)

Determine if the provider supports Responses API or requires Completion API:

```python
# In ModelRoute._finalize_model_route_object()
self.is_target_anthropic = self.target_model.startswith(f"{ANTHROPIC}/")
self.is_target_mistral = self.target_model.startswith("mistral/")

if self.is_target_anthropic:
    self.use_responses_api = False
elif self.is_target_mistral:
    # Mistral doesn't support Responses API
    self.use_responses_api = False
else:
    self.use_responses_api = ALWAYS_USE_RESPONSES_API or ...
```

**How to test:** Check LiteLLM logs for which API is called:
- Responses API: `litellm.aresponses()` 
- Completion API: `litellm.acompletion()`

### 2. Message Field Stripping (`provider_transforms.py`)

Anthropic-specific fields that other providers may reject:

| Field | Error Message | Solution |
|-------|---------------|----------|
| `thinking_blocks` | "Extra inputs are not permitted" | Strip from all messages |
| `metadata` | "unsupported parameter metadata" | Strip from params (Responses API) |
| `context_management` | Various | Strip from params |

```python
# In provider_transforms.py
_REQUEST_RULES.append(
    ProviderRule(
        name="strip_thinking_blocks_non_anthropic",
        when=lambda ctx: (not ctx.model_route.is_target_anthropic),
        apply=lambda ctx: _strip_thinking_blocks_from_messages(ctx.messages),
    )
)
```

### 3. Tool Call/Response Pairing

Some providers (like Mistral) require strict pairing between tool_calls and tool responses.

**Mistral errors:**
- "Not the same number of function calls and responses"
- "Unexpected role 'tool' after role 'system'"

**Solution:** The `_fix_tool_call_response_pairing()` function:
1. Collects all tool_call IDs from assistant messages
2. Collects tool_call IDs that have matching responses
3. Removes orphaned tool_calls (no response) and orphaned tool responses (no call)
4. Ensures tool messages only follow assistant messages with tool_calls

```python
def _apply_mistral_fixes(ctx) -> None:
    ctx.messages[:] = _fix_tool_call_response_pairing(ctx.messages)

_REQUEST_RULES.append(
    ProviderRule(
        name="mistral_tool_pairing",
        when=lambda ctx: ctx.model_route.target_model.startswith("mistral/"),
        apply=_apply_mistral_fixes,
    )
)
```

### 4. Message Order Constraints

Some providers have strict message ordering rules:

| Provider | Constraint |
|----------|------------|
| Mistral | No `system` after `tool` |
| Mistral | `tool` must immediately follow `assistant` with `tool_calls` |

**Solution:** Narrow the system prompt injection rule for strict providers:

```python
_REQUEST_RULES.append(
    ProviderRule(
        name="inject_system_prompt_instructions",
        when=lambda ctx: (not ctx.model_route.is_target_mistral),
        apply=_inject_system_prompt_instructions,
    )
)
```

## Debugging Steps

### Step 1: Identify the API Path

Check logs for which method is called:
```
mistral/devstral-2512 -> mistral/devstral-2512
```

If it shows the wrong API (e.g., `aresponses` instead of `acompletion`):
- Check `use_responses_api` logic in `route_model.py`
- Check `ALWAYS_USE_RESPONSES_API` environment variable

### Step 2: Identify Rejected Fields

Common error patterns:
```
MistralException - {"detail":[{"type":"extra_forbidden","loc":["body","messages",X,"assistant","thinking_blocks"]...
```

Parse the error:
- `loc` shows the path to the rejected field
- `type` shows the rejection reason

### Step 3: Check Message Order

Errors like "Unexpected role 'X' after role 'Y'" indicate:
- Orphaned tool responses
- System messages in wrong position
- Missing assistant messages with tool_calls

Add debug logging:
```python
print(f"[DEBUG] Skipping tool msg: tc_id={tc_id}, prev_role={prev_role}")
```

### Step 4: Clear Bytecode Cache

After making changes:
```bash
rm -rf claude_code_proxy/__pycache__
rm -rf common/__pycache__
# Restart LiteLLM proxy
```

Or set in startup script:
```bash
export PYTHONDONTWRITEBYTECODE=1
```

## Testing Approach

1. **Minimal test:** Simple "hi" message without tool calls
2. **Tool call test:** Trigger a tool call and response
3. **Multi-turn test:** Extended conversation with multiple tool exchanges
4. **Error recovery:** Test after a failed tool call

## Example: Adding a New Provider

```python
# 1. In route_model.py - Add API mode check
self.is_target_newprovider = self.target_model.startswith("newprovider/")

if self.is_target_newprovider:
    self.use_responses_api = False  # or True based on capability

# 2. In provider_transforms.py - Add provider-specific rules
def _strip_newprovider_fields(messages: list) -> None:
    for msg in messages:
        if isinstance(msg, dict):
            msg.pop("provider_specific_field", None)

_REQUEST_RULES.append(
    ProviderRule(
        name="newprovider_strip_field",
        when=lambda ctx: ctx.model_route.target_model.startswith("newprovider/"),
        apply=lambda ctx: _strip_newprovider_fields(ctx.messages),
    )
)

def _apply_newprovider_tool_pairing(ctx) -> None:
    ctx.messages[:] = _fix_tool_call_response_pairing(ctx.messages)

_REQUEST_RULES.append(
    ProviderRule(
        name="newprovider_tool_pairing",
        when=lambda ctx: ctx.model_route.target_model.startswith("newprovider/"),
        apply=_apply_newprovider_tool_pairing,
    )
)

# 3. If strict ordering, skip system prompt injection by narrowing the rule
# (See provider_transforms.py: "inject_system_prompt_instructions")
```

## LiteLLM Config Example

```yaml
model_list:
  - model_name: my-model-group
    litellm_params:
      # Route through custom handler
      model: claude_code_router/newprovider/model-name
      api_base: https://api.newprovider.com/v1
      api_key: os.environ/NEWPROVIDER_API_KEY
```

## Common Pitfalls

1. **Bytecode caching:** Changes not taking effect → clear `__pycache__`
2. **Environment variables:** `ALWAYS_USE_RESPONSES_API=true` overrides per-provider settings
3. **LiteLLM callbacks:** `async_pre_call_hook` runs BEFORE router selects deployment
4. **`additional_drop_params`:** Only works for some providers, not Mistral native handler
5. **Nested field paths:** Use handler transformation, not `drop_params` for message fields

## Related Documentation

- [custom-upstream-openai-responses.md](./custom-upstream-openai-responses.md) - OpenAI Responses API specifics
- [LiteLLM Custom Providers](https://docs.litellm.ai/docs/providers/custom_provider)
