"""Shared helpers for offline transcription and benchmarking entrypoints."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from kdictate.app_metadata import DEFAULT_MODEL_DIR
from kdictate.audio_common import load_whisper_model
from kdictate.runtime_profile import (
    recommended_shortform_cpu_threads,
    resolve_runtime,
    set_thread_env,
)


def add_shared_runtime_args(parser: argparse.ArgumentParser) -> None:
    """Add the common CPU/runtime arguments used by offline entrypoints."""

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to converted CTranslate2 model directory.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("auto", "cpu"),
        help="Inference device (CPU only in this project).",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        choices=("float32", "float16", "int8", "int8_float16"),
        help="faster-whisper compute type. If omitted, auto-selects.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=recommended_shortform_cpu_threads(),
        help="Override CPU thread count. Defaults to a short-form latency-oriented value.",
    )
    parser.add_argument("--language", default="en", help="Language code (for example: en, es, fr).")


def resolve_input_paths(audio: str, model_dir: str) -> tuple[Path, Path]:
    """Return validated audio/model paths or raise FileNotFoundError."""

    audio_path = Path(audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    return audio_path, model_path


def load_offline_model(
    model_dir: Path,
    *,
    device: str | None,
    compute_type: str | None,
    cpu_threads: int | None,
) -> tuple[Any, dict[str, Any], float]:
    """Resolve runtime settings and load the shared Whisper model."""

    runtime = resolve_runtime(device, compute_type, cpu_threads)
    set_thread_env(runtime["cpu_threads"])
    load_start = time.perf_counter()
    model = load_whisper_model(
        model_dir,
        device=runtime["device"],
        compute_type=runtime["compute_type"],
        cpu_threads=runtime["cpu_threads"],
    )
    return model, runtime, time.perf_counter() - load_start
