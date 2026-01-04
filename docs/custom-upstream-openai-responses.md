# Custom upstream OpenAI provider integration guide

## Scope
This guide records the integration fixes and development context for using `claude_code_router` with a custom upstream OpenAI compatible provider via LiteLLM.

## Integration overview
- LiteLLM loads the custom provider from `litellm_settings.custom_provider_map`.
- The handler adapts Claude Code requests and routes to OpenAI style models.
- When using Responses API, LiteLLM does not honor `drop_params`.

## Fixes implemented in this repo
1) **Forward auth and base URL**
   - The handler now forwards `api_base` and `api_key` into all downstream `litellm.responses|completion` calls.
   - File: `claude_code_proxy/claude_code_router.py`

2) **Config driven metadata stripping for Responses API**
   - Added a flag `drop_metadata_res_api` that removes `metadata` from the Responses API payload for models that opt in.
   - The handler checks this flag in `litellm_params`, `optional_params`, and `optional_params.litellm_params` to be robust to LiteLLM propagation.
   - File: `claude_code_proxy/claude_code_router.py`

3) **Config driven thinking blocks stripping for Responses API**
   - Added a flag `drop_thinking_blocks_res_api` that removes `thinking_blocks` from Responses API input items for models that opt in.
   - The handler checks this flag in `litellm_params`, `optional_params`, and `optional_params.litellm_params` to be robust to LiteLLM propagation.
   - File: `claude_code_proxy/claude_code_router.py`

4) **Preserve thinking_blocks for Anthropic-compatible upstreams**
   - For non-`anthropic/` prefixed models that route to Anthropic-compatible APIs (e.g., packyapi.com), thinking_blocks must be preserved.
   - Add `preserve_thinking_blocks` to `additional_drop_params` to skip thinking_blocks stripping.
   - The handler checks `litellm_params.additional_drop_params` for this flag.
   - File: `claude_code_proxy/claude_code_router.py`

## Example LiteLLM model config
```yaml
model_list:
  - model_name: custom-openai/gpt-5.2-high
    litellm_params:
      model: claude_code_router/gpt-5.2-high
      api_base: https://custom-openai.example.com/v1
      api_key: os.environ/CUSTOM_OPENAI_API_KEY
      drop_metadata_res_api: true
      drop_thinking_blocks_res_api: true
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
  - Set `drop_metadata_res_api: true` for the affected model.
  - Confirm the generated config contains the flag used at runtime.
- **400 unsupported parameter thinking_blocks**
  - Set `drop_thinking_blocks_res_api: true` for the affected model.
  - Confirm the generated config contains the flag used at runtime.
- **Module import errors**
  - Confirm the config directory contains `claude_code_proxy/` and `common/`.
  - Confirm `PYTHONPATH` includes the config directory.

## Mistral-specific fixes

Mistral has strict message ordering requirements and doesn't support Responses API. The following fixes were implemented:

1. **Force Completion API for Mistral** (`route_model.py`)
   - `use_responses_api = False` for all `mistral/*` models
   - Ignores `ALWAYS_USE_RESPONSES_API` env var for Mistral

2. **Strip thinking_blocks** (`claude_code_router.py`)
   - Removed from all messages in `_adapt_complapi_for_non_anthropic_models()`
   - Error: "Extra inputs are not permitted" at `messages[*].assistant.thinking_blocks`
   - **Exception:** If `preserve_thinking_blocks` is in `additional_drop_params`, thinking_blocks are preserved (for Anthropic-compatible upstreams like packyapi.com)

3. **Fix tool call/response pairing** (`claude_code_router.py`)
   - `_fix_tool_call_response_pairing()` removes orphaned tool_calls and tool responses
   - Error: "Not the same number of function calls and responses"

4. **Skip system message injection** (`claude_code_router.py`)
   - Mistral doesn't allow system messages after tool messages
   - Error: "Unexpected role 'system' after role 'tool'"

5. **Ensure tool message order**
   - Tool messages must immediately follow assistant messages with matching tool_calls
   - Error: "Unexpected role 'tool' after role 'system'"

## Future changes
- If upstream starts accepting `metadata`, remove the flag to restore metadata logging.
- If upstream starts accepting `thinking_blocks`, remove the flag to preserve reasoning signatures.
- If another field is rejected by Responses API, add a new opt in flag and strip it in the handler.
- If LiteLLM changes custom handler loading, revisit the config directory and `PYTHONPATH` requirements.

## Related documentation
- [Adding New Providers](./adding-new-providers.md) - Comprehensive guide for integrating new providers
