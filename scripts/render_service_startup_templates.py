"""Render service startup env and systemd templates."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.service_startup_templates import (  # noqa: E402
    DEFAULT_CHUNK_POLICY_NAMES,
    build_service_startup_template_plan,
    service_startup_template_plan_payload,
    write_service_startup_templates,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render NeX_PCX app-host service startup templates.",
    )
    parser.add_argument("--workdir", default=str(PROJECT_ROOT))
    parser.add_argument("--user", default="nexpcx")
    parser.add_argument("--group", default=None)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8000)
    parser.add_argument(
        "--database-url-placeholder",
        default="postgresql://<user>:<password>@127.0.0.1:5432/nex_pcx_app",
    )
    parser.add_argument("--upload-storage-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--environment-name", default="production")
    parser.add_argument("--restart-seconds", type=int, default=5)
    parser.add_argument(
        "--chunk-policy-names",
        nargs="+",
        default=list(DEFAULT_CHUNK_POLICY_NAMES),
    )
    parser.add_argument(
        "--user-systemd",
        action="store_true",
        help=(
            "Render user-level systemd units. User/group and system-only "
            "hardening directives are omitted, and units install to default.target."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write env/systemd/readme files. Without this flag the command prints JSON only.",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    plan = build_service_startup_template_plan(
        workdir=args.workdir,
        user=args.user,
        group=args.group,
        python_bin=args.python_bin,
        output_dir=args.output_dir,
        web_host=args.web_host,
        web_port=args.web_port,
        database_url_placeholder=args.database_url_placeholder,
        upload_storage_dir=args.upload_storage_dir,
        models_dir=args.models_dir,
        environment_name=args.environment_name,
        restart_seconds=args.restart_seconds,
        chunk_policy_names=tuple(args.chunk_policy_names),
        user_systemd=args.user_systemd,
    )
    written_files = write_service_startup_templates(plan) if args.write else ()
    payload = {
        **service_startup_template_plan_payload(plan),
        "wrote_files": bool(written_files),
        "written_files": [str(path) for path in written_files],
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_text + "\n", encoding="utf-8")
    else:
        print(json_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
