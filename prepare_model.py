from __future__ import annotations

"""Convert Whisper model weights to local CTranslate2 format.

This is a one-time prep step that downloads model assets and writes a local
directory optimized for `faster-whisper`.
"""

import argparse
import inspect
import shutil
import sys
from pathlib import Path

from kdictate.app_metadata import DEFAULT_MODEL_DIR, DEFAULT_MODEL_ID

REQUIRED_METADATA_FILES = ("tokenizer.json", "preprocessor_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Whisper-compatible model to local CTranslate2 format.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID to convert.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Destination directory for converted model.",
    )
    parser.add_argument(
        "--quantization",
        default="int8",
        choices=("float32", "float16", "int16", "int8", "int8_float16"),
        help="Quantization type used in CTranslate2 conversion.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    return parser.parse_args()


def _copy_metadata_files(model_id: str, output_dir: Path) -> None:
    """Ensure tokenizer/preprocessing files exist for runtime consumers.

    Older CTranslate2 converter builds do not support `copy_files`, so we
    manually download these small metadata files when needed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Warning: huggingface-hub not available; metadata copy skipped.")
        return

    for filename in REQUIRED_METADATA_FILES:
        target_path = output_dir / filename
        if target_path.exists():
            continue
        try:
            source_path = hf_hub_download(repo_id=model_id, filename=filename)
            shutil.copy2(source_path, target_path)
            print(f"Copied metadata file: {filename}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not copy {filename}: {exc}")


def _ensure_torch_available() -> bool:
    """Fail early with actionable guidance when torch is unavailable.

    CTranslate2's Transformers converter needs PyTorch to load HF weights.
    Some converter versions surface this as a runtime NameError, so we check
    before conversion to provide a clear fix.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Missing dependency: torch")
        print("Model conversion requires PyTorch.")
        print("Install project dependencies in this venv, then rerun:")
        print("  pip install -r requirements.txt")
        if sys.version_info >= (3, 14):
            print("If no torch wheel exists for Python 3.14, use a Python 3.12 venv for conversion.")
        return False
    return True


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    # Ensure the parent path exists before conversion starts.
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        print("Missing dependency: ctranslate2")
        print("Install dependencies first: pip install -r requirements.txt")
        return 1

    if not _ensure_torch_available():
        return 1

    print(f"Converting {args.model_id} -> {output_dir} ({args.quantization})")
    converter = TransformersConverter(args.model_id)
    convert_kwargs = {
        "output_dir": str(output_dir),
        "quantization": args.quantization,
        "force": args.force,
    }

    # Support both old and new converter APIs.
    signature = inspect.signature(converter.convert)
    if "copy_files" in signature.parameters:
        convert_kwargs["copy_files"] = list(REQUIRED_METADATA_FILES)
    try:
        converter.convert(**convert_kwargs)
    except NameError as exc:
        # Older converter code can throw NameError("torch") instead of ImportError.
        if "torch" in str(exc):
            print("Conversion failed because PyTorch is not available in this environment.")
            print("Install project dependencies, then rerun:")
            print("  pip install -r requirements.txt")
            if sys.version_info >= (3, 14):
                print("If no torch wheel exists for Python 3.14, use a Python 3.12 venv for conversion.")
            return 1
        raise

    if "copy_files" not in signature.parameters:
        _copy_metadata_files(args.model_id, output_dir)

    print("Conversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
