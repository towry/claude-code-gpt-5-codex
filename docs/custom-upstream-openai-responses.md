# Custom upstream OpenAI provider integration guide

## Scope
This guide records the integration fixes and development context for using `claude_code_router` with a custom upstream OpenAI compatible provider via LiteLLM.

## Integration overview
- LiteLLM loads the custom provider from `litellm_settings.custom_provider_map`.
- The handler adapts Claude Code requests and routes to OpenAI style models.
- When using Responses API, LiteLLM does not honor `drop_params`.

## Fixes implemented in this repo
1) **Forward auth and base URL**
   - The handler forwards `api_base` and `api_key` into all downstream `litellm.responses|completion` calls.
   - File: `claude_code_proxy/claude_code_router.py`

2) **Provider transform pipeline**
   - Request/response adjustments are centralized in a rule-based pipeline for easy provider extensions.
   - File: `claude_code_proxy/provider_transforms.py`

3) **Responses API metadata stripping**
   - `metadata` is always removed from Responses API payloads (OpenAI Responses rejects Anthropic metadata).
   - File: `claude_code_proxy/provider_transforms.py`

4) **Thinking block handling**
   - `preserve_thinking_blocks` (in `additional_drop_params`) skips default `thinking_blocks` stripping for Anthropic-compatible upstreams.
   - `strip_thinking_blocks` (in `additional_drop_params`) removes `thinking`/`redacted_thinking` blocks from requests and responses to prevent invalid signatures.
   - `strip_thinking_blocks` overrides `preserve_thinking_blocks`.
   - File: `claude_code_proxy/provider_transforms.py`

## Example LiteLLM model config
```yaml
model_list:
  - model_name: custom-openai/gpt-5.2-high
    litellm_params:
      model: claude_code_router/gpt-5.2-high
      api_base: https://custom-openai.example.com/v1
      api_key: os.environ/CUSTOM_OPENAI_API_KEY
      additional_drop_params:
        - strip_thinking_blocks
```

## Handler loading requirements
LiteLLM loads custom handlers relative to the config directory.

Required setup:
- Ensure `claude_code_proxy/` and `common/` are present in the config directory.
- Ensure the config directory is on `PYTHONPATH`.

## Troubleshooting
- **401 auth errors**
  - Verify `api_base` and `api_key` are present in the model `litellm_params`.
  - Confirm the handler forwards these values into downstream calls.
- **400 unsupported parameter metadata**
  - Metadata is removed for Responses API payloads by default.
- **400 unsupported parameter thinking_blocks**
  - Thinking blocks are removed for Responses API inputs by default.
- **Invalid signature when switching to official Anthropic**
  - Set `strip_thinking_blocks` in `additional_drop_params` for the Anthropic-compatible upstream (and for the official model if you need to sanitize existing history).
- **Module import errors**
  - Confirm the config directory contains `claude_code_proxy/` and `common/`.
  - Confirm `PYTHONPATH` includes the config directory.

## Mistral-specific fixes

Mistral has strict message ordering requirements and doesn't support Responses API. The following fixes were implemented:

1. **Force Completion API for Mistral** (`route_model.py`)
   - `use_responses_api = False` for all `mistral/*` models
   - Ignores `ALWAYS_USE_RESPONSES_API` env var for Mistral

2. **Strip thinking_blocks** (`provider_transforms.py`)
   - Removed from all messages in the request transform pipeline
   - Error: "Extra inputs are not permitted" at `messages[*].assistant.thinking_blocks`
   - **Exception:** If `preserve_thinking_blocks` is in `additional_drop_params`, thinking_blocks are preserved (for Anthropic-compatible upstreams like packyapi.com)

3. **Fix tool call/response pairing** (`provider_transforms.py`)
   - `_fix_tool_call_response_pairing()` removes orphaned tool_calls and tool responses
   - Error: "Not the same number of function calls and responses"

4. **Skip system message injection** (`provider_transforms.py`)
   - Mistral doesn't allow system messages after tool messages
   - Error: "Unexpected role 'system' after role 'tool'"

5. **Ensure tool message order**
   - Tool messages must immediately follow assistant messages with matching tool_calls
   - Error: "Unexpected role 'tool' after role 'system'"

## Future changes
- If upstream starts accepting `metadata`, relax the Responses API stripping rule.
- If upstream starts accepting `thinking_blocks`, remove `strip_thinking_blocks` or add `preserve_thinking_blocks`.
- If another field is rejected by Responses API, add a new rule to the transform pipeline.
- If LiteLLM changes custom handler loading, revisit the config directory and `PYTHONPATH` requirements.

## Related documentation
- [Adding New Providers](./adding-new-providers.md) - Comprehensive guide for integrating new providers
