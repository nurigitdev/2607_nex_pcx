# Generation Provider Runtime Config UI

Slice 350 adds an operator page for reviewing and seeding generation provider
runtime settings.

## Page

`GET /admin/generation-provider-configs`

The page shows:

- total provider count
- active provider count
- current default provider
- default provider API key environment variable reference
- provider mode and remote base URL
- model id
- token, timeout, temperature, and top_p limits
- runtime validation status
- API key environment configuration status for the known DGX key
- redacted raw config JSON evidence

## DGX Seed Action

`POST /admin/generation-provider-configs/seed-dgx-vllm`

The form upserts the DGX Qwen3.5 122B vLLM runtime defaults without accepting or
persisting an API key value. Operators can explicitly choose whether the seeded
remote provider should become the default.

The form uses these defaults:

- `dgx_vllm_qwen35_122b_a10b_nvfp4`
- `http://192.168.20.243:12000`
- `/home/nurivoice-dgx/models/nvidia/Qwen3.5-122B-A10B-NVFP4`
- `NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY`
- Qwen thinking disabled through `extra_body`

## Navigation

The page is linked from the left navigation as Generation Provider Config and
from the mobile navigation with the same label.

## Acceptance Checks

- DB-disabled mode shows an operator warning and disables the seed form.
- Seed success redirects back to the page with the saved provider name.
- Runtime config payloads are rendered from the same API serialization helper.
- Secret values are not displayed; only the environment variable name appears.
- Korean is the default UI language and English fallback text is available.
