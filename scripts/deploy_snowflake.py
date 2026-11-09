"""
Empaqueta el paquete `churn/` y sube `churn.zip` + `config.yaml` al stage,
para que el Stored Procedure use siempre el mismo código que el entrenamiento.

    python scripts/deploy_snowflake.py --config config/config.yaml
"""
import argparse
import os
import sys
import tempfile
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from churn import io  # noqa: E402
from churn.config import load_config  # noqa: E402


def build_zip(dest: str) -> str:
    path = os.path.join(dest, "churn.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(os.path.join(ROOT, "churn"))):
            if fname.endswith(".py"):
                zf.write(os.path.join(ROOT, "churn", fname), arcname=f"churn/{fname}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config, overrides={"mode": "snowflake"})
    session = io.create_snowflake_session(cfg)
    try:
        session.sql(f"CREATE STAGE IF NOT EXISTS {cfg.stage_uri[1:]}").collect()
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = build_zip(tmp)
            cfg_copy = os.path.join(tmp, "config.yaml")
            with open(args.config, "rb") as src, open(cfg_copy, "wb") as dst:
                dst.write(src.read())
            for path in (zip_path, cfg_copy):
                session.file.put(path, cfg.stage_uri, auto_compress=False, overwrite=True)
                print(f"[OK] Subido: {cfg.stage_uri}/{os.path.basename(path)}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
