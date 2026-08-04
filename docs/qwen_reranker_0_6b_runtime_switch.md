# Qwen3 Reranker 0.6B Runtime Switch

This slice changes the active NeX-PCX reranker contract from
`Qwen/Qwen3-Reranker-4B` to `Qwen/Qwen3-Reranker-0.6B`.

## Runtime Contract

| Item | Value |
| --- | --- |
| Provider URL | `http://192.168.20.243:9104` |
| Reranker profile | `qwen3_reranker_0_6b` |
| Reranker model id | `Qwen/Qwen3-Reranker-0.6B` |
| DGX model directory | `/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_0_6b` |
| Provider backend | `qwen_reranker` |
| Recommended dtype | `bfloat16` |

The app-side expected profile and model id can be overridden without code
changes:

```bash
export NEX_PCX_RERANKER_PROFILE_NAME="qwen3_reranker_0_6b"
export NEX_PCX_RERANKER_MODEL_ID="Qwen/Qwen3-Reranker-0.6B"
```

The DGX provider service uses a separate provider-side contract:

```bash
NEX_PCX_RERANKER_PROVIDER_MODEL_DIR_NAME=qwen3_reranker_0_6b
NEX_PCX_RERANKER_PROVIDER_MODEL_ID=Qwen/Qwen3-Reranker-0.6B
NEX_PCX_RERANKER_PROVIDER_PROFILE_NAME=qwen3_reranker_0_6b
```

## Operator Sequence

1. Copy or download the model bundle to
   `/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_0_6b` on DGX.
2. Update
   `/home/nexpcx/2607_nex_pcx/deployment/env/nex-pcx-reranker-provider.env`
   with the provider-side contract above.
3. Restart the DGX user service:

   ```bash
   ssh nexpcx@192.168.20.243 "systemctl --user restart nex-pcx-reranker-provider.service"
   ```

4. Check the provider contract:

   ```bash
   curl -fsS http://192.168.20.243:9104/healthz
   ./.venv/bin/python scripts/run_remote_reranker_request_smoke.py \
     --base-url http://192.168.20.243:9104 \
     --markdown-output artifacts/remote_reranker_request_smoke.md
   ```

5. Restart the NeX-PCX app process so the app-side settings are reloaded.
6. Confirm `/api/admin/reranker-provider/status?request_smoke=true` reports the
   same model id and profile.

## DGX Switch Evidence

The model bundle was downloaded to
`/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_0_6b` on 2026-08-04. The
directory size is approximately `1.2G` and includes `model.safetensors`,
`config.json`, and tokenizer files.

The DGX user systemd env file was updated:

```bash
NEX_PCX_RERANKER_PROVIDER_MODEL_DIR_NAME=qwen3_reranker_0_6b
NEX_PCX_RERANKER_PROVIDER_MODEL_ID=Qwen/Qwen3-Reranker-0.6B
NEX_PCX_RERANKER_PROVIDER_PROFILE_NAME=qwen3_reranker_0_6b
```

After `systemctl --user restart nex-pcx-reranker-provider.service`, `/healthz`
reported:

```json
{
  "ready": true,
  "provider_model_id": "Qwen/Qwen3-Reranker-0.6B",
  "reranker_profile_name": "qwen3_reranker_0_6b",
  "device": "cuda:0",
  "runtime_metadata": {
    "model_dir_exists": true,
    "loaded_parameter_dtype": "bfloat16"
  }
}
```

The request smoke passed and wrote
`artifacts/remote_reranker_0_6b_request_smoke.md`. Post-smoke systemd status
reported approximately `1.1G` resident memory, down from the previous 4B
service observation of approximately `8.8G`.
