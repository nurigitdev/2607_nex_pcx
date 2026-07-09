# Embedding Models

This directory is the default local model bundle root for NeX_PCX.

Downloaded model files are intentionally ignored by git. Use
`scripts/download_embedding_models.py` to populate subdirectories such as:

- `models/kure_v1`
- `models/bge_m3`
- `models/qwen3_embedding_4b`

Production and customer-site deployments should receive a verified copy of
this directory instead of downloading models during application startup.
