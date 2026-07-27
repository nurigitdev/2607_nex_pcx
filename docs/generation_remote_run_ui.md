# Remote Generation Run UI Controls

Slice 353 adds remote vLLM execution controls to the generation screen.

## Page

`GET /generation`

When a retrieval context package is loaded, the page now shows a
`Remote vLLM Runtime` panel next to the generation actions. The panel displays:

- default generation provider name
- provider mode
- model id
- API key environment readiness
- request timeout
- remote execution readiness badge

The `Remote vLLM Generate` action is disabled unless the default provider runtime
is valid, uses `remote_openai_compatible`, and any configured API key environment
reference is available to the running process.

## Form Action

`POST /generation/runs/remote`

The form handler mirrors the API path. It builds the retrieval package, resolves
the default provider API key from `runtime_options.api_key_env`, calls
`execute_remote_generation_run(...)`, and redirects back to `/generation` with
the new `generation_run_id` selected.

Remote provider failures are shown as persisted generation runs. Local readiness
or configuration errors redirect back to the page with a warning message.

## Language

Korean remains the default UI language. English strings are available for the
same controls through the existing locale bundle.
