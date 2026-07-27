# DGX vLLM Generation Run Live E2E Result

- `passed`: `true`
- `provider_name`: `slice_355_dgx_vllm_generation_e2e`
- `provider_base_url`: `http://192.168.20.243:12000`
- `model_id`: `/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4`
- `api_key_env`: `NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY`
- `api_key_configured`: `true`
- `request_timeout_seconds`: `300`
- `max_tokens`: `512`
- `temperature`: `0`
- `top_p`: `1`
- `search_log_id`: `25`
- `generation_run_id`: `1`
- `status`: `succeeded`
- `guardrail_status`: `allowed`
- `retrieval_confidence_status`: `answerable`
- `citation_readiness_status`: `warning`
- `citation_count`: `1`
- `cited_count`: `1`
- `finish_reason`: `stop`
- `input_token_count`: `363`
- `output_token_count`: `18`
- `total_token_count`: `381`
- `elapsed_ms`: `1054`
- `provider_elapsed_ms`: `1054`
- `provider_http_status_code`: `200`
- `provider_response_id`: `chatcmpl-9f69e1246f5ff601`
- `cleanup_confirmed`: `true`
- `default_provider_restored`: `true`

## Answer Preview

회사 보안 규정은 계정 공유를 금지합니다 [RCP-001].

## Provider Metrics

```json
{
  "raw_usage": {
    "total_tokens": 381,
    "prompt_tokens": 363,
    "completion_tokens": 18,
    "prompt_tokens_details": null
  },
  "succeeded": true,
  "elapsed_ms": 1054,
  "error_code": null,
  "response_id": "chatcmpl-9f69e1246f5ff601",
  "retry_count": 0,
  "error_message": null,
  "finish_reason": "stop",
  "provider_mode": "remote_openai_compatible",
  "provider_name": "slice_355_dgx_vllm_generation_e2e",
  "contract_version": "generation_provider_metrics_v1",
  "http_status_code": 200,
  "input_token_count": 363,
  "response_metadata": {
    "object": "chat.completion",
    "created": 1785117952,
    "choice_count": 1,
    "service_tier": null,
    "system_fingerprint": "vllm-0.23.0-cf09107d"
  },
  "response_model_id": "/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4",
  "total_token_count": 381,
  "output_token_count": 18,
  "requested_model_id": "/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4",
  "provider_elapsed_ms": 1054
}
```

## Fixture

- `smoke_run_key`: `slice-355-dgx-vllm-generation-e2e-a105db0f-a2ab-4f66-be0b-22175821e3dd`
- `file_id`: `54`
- `document_id`: `54`
- `chunk_id`: `96`
