# DGX vLLM Generation Smoke Result

- `passed`: `true`
- `provider_name`: `dgx-vllm-qwen3-6-27b`
- `chat_completions_url`: `http://192.168.20.243:12000/v1/chat/completions`
- `model_id`: `/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4`
- `api_key_env`: `NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY`
- `api_key_configured`: `true`
- `serving_max_model_len`: `200k`
- `timeout_seconds`: `300`
- `max_tokens`: `96`
- `temperature`: `0`
- `top_p`: `1`
- `thinking_disabled`: `true`
- `http_status_code`: `200`
- `response_model_id`: `/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4`
- `response_id`: `chatcmpl-bc64978293d4ac02`
- `finish_reason`: `stop`
- `request_elapsed_ms`: `578`
- `provider_elapsed_ms`: `577`
- `input_token_count`: `77`
- `output_token_count`: `12`
- `total_token_count`: `89`
- `answer_char_count`: `21`

## Answer Preview

NeX-PCX vLLM 연결 확인 완료

## Provider Metrics

```json
{
  "contract_version": "generation_provider_metrics_v1",
  "provider_name": "dgx-vllm-qwen3-6-27b",
  "provider_mode": "remote_openai_compatible",
  "requested_model_id": "/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4",
  "response_model_id": "/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4",
  "response_id": "chatcmpl-bc64978293d4ac02",
  "http_status_code": 200,
  "finish_reason": "stop",
  "input_token_count": 77,
  "output_token_count": 12,
  "total_token_count": 89,
  "elapsed_ms": 577,
  "provider_elapsed_ms": 577,
  "retry_count": 0,
  "succeeded": true,
  "error_code": null,
  "error_message": null,
  "raw_usage": {
    "prompt_tokens": 77,
    "total_tokens": 89,
    "completion_tokens": 12,
    "prompt_tokens_details": null
  },
  "response_metadata": {
    "object": "chat.completion",
    "created": 1785107874,
    "system_fingerprint": "vllm-0.23.0-cf09107d",
    "service_tier": null,
    "choice_count": 1
  }
}
```
