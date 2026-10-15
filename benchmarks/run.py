"""Compare CLI and resident inference, retaining every hypothesis and timing.

WER: lowercase Unicode words, normalize apostrophes, remove punctuation;
numbers remain literal. Report micro-averaged edit distance, not averaged
per-clip percentages. Empty references are separate hallucination checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from kdictate.backend import WhisperCppBackend
from kdictate.audio_common import postprocess_transcript
from kdictate.resident_backend import ResidentBackend


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.lower().replace("’", "'"))


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        next_row = [i]
        for j, other in enumerate(hypothesis, 1):
            next_row.append(min(row[j] + 1, next_row[j-1] + 1, row[j-1] + (word != other)))
        row = next_row
    return row[-1]


def pcm(path):
    with wave.open(path) as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError("benchmark audio must be mono s16le at 16 kHz")
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").copy()


def process_memory(pid: int) -> dict[str, int]:
    """Linux RSS plus per-client DRM allocations when the driver exposes them."""
    status = Path(f"/proc/{pid}/status").read_text()
    result = {key: int(value) for key, value in re.findall(r"^(VmRSS|VmHWM):\s+(\d+) kB", status, re.M)}
    clients = set()
    for path in Path(f"/proc/{pid}/fdinfo").iterdir():
        try:
            fields = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
        except OSError:
            continue
        if "drm-client-id" not in fields:
            continue
        client = (fields.get("drm-pdev", ""), fields["drm-client-id"])
        if client in clients:
            continue
        clients.add(client)
        for key, value in fields.items():
            if key.startswith(("drm-memory-", "drm-total-", "drm-resident-")):
                match = re.fullmatch(r"\s*(\d+) (?:KiB|kB)\s*", value)
                if match:
                    result[key] = result.get(key, 0) + int(match[1])
    return result


def worker_memory(backend: ResidentBackend) -> dict[str, int] | None:
    """A failed request may already have reaped the worker."""
    process = backend._process
    if process is None or process.poll() is not None:
        return None
    try:
        return process_memory(process.pid)
    except OSError:
        # The worker can exit between poll() and reading /proc.
        return None


def load_manifest(path: Path) -> dict:
    """Resolve portable audio paths against the manifest, independent of cwd."""
    manifest = json.loads(path.read_text())
    for sample in manifest["samples"]:
        audio = Path(sample["path"])
        if not audio.is_absolute():
            audio = (path.parent / audio).resolve()
        sample["path"] = str(audio)
    return manifest


def vulkan_summary() -> str:
    try:
        info = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    return "\n".join(line.strip() for line in info.stdout.splitlines()
                     if any(key in line for key in ("deviceName", "driverName", "driverInfo")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", choices=("whisper", "parakeet"), default="whisper")
    parser.add_argument("--mode", choices=("cli", "resident"), default="cli")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--repeat and --limit must be positive")
    manifest = load_manifest(args.manifest)
    samples = manifest["samples"][:args.limit] if args.limit else manifest["samples"]
    for sample in samples:
        if hashlib.sha256(Path(sample["path"]).read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError(f"audio checksum mismatch: {sample['id']}")
    with open(args.model, "rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    result = dict(name=args.name, mode=args.mode, engine=args.engine, binary=str(Path(args.binary).resolve()),
                  binary_sha256=hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(), model=args.model,
                  model_sha256=model_sha256, dataset=manifest["dataset"], dataset_revision=manifest["revision"],
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  started=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                  vulkan=vulkan_summary(), environment={k: v for k, v in os.environ.items() if k.startswith(("GGML_", "VK_", "AMD_VULKAN_"))},
                  cpu=next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
                  load_start=Path("/proc/loadavg").read_text().strip(),
                  memory_start=Path("/proc/meminfo").read_text(),
                  threads=6, beam=3 if args.engine == "whisper" else "greedy", rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    backend = None
    try:
        started = time.perf_counter()
        if args.mode == "resident":
            backend = ResidentBackend(args.binary, args.model, engine=args.engine)
            result["startup_seconds"] = time.perf_counter() - started
            if not args.no_warmup:
                started = time.perf_counter()
                backend.transcribe([np.zeros(16000, dtype=np.int16)], 1.0)
                result["warmup_seconds"] = time.perf_counter() - started
            result["idle_memory_kib"] = worker_memory(backend)
        elif args.engine == "whisper":
            backend = WhisperCppBackend(args.binary, args.model)
        for repeat in range(args.repeat):
            for sample in samples:
                audio = pcm(sample["path"])
                started = time.perf_counter()
                try:
                    if backend is not None:
                        text = backend.transcribe([audio], sample["seconds"])
                    else:
                        completed = subprocess.run([args.binary, "--model", args.model, "--threads", "6",
                                                    "--no-prints", "--file", sample["path"]],
                                                   capture_output=True, timeout=90, check=True)
                        text = postprocess_transcript(completed.stdout.decode().strip())
                    error = None
                except Exception as exc:
                    text, error = "", str(exc)
                seconds = time.perf_counter() - started
                reference, hypothesis = words(sample["reference"]), words(text)
                row = dict(id=sample["id"], group=sample["group"], speaker=sample.get("speaker"), audio_seconds=sample["seconds"],
                           reference=sample["reference"], hypothesis=text, elapsed_seconds=seconds,
                           reference_words=len(reference), edits=edit_distance(reference, hypothesis),
                           error=error, repeat=repeat)
                result["rows"].append(row)
                print(f"{args.name} {len(result['rows'])}/{len(samples)*args.repeat} {row['id']} {seconds:.2f}s {row['edits']}/{len(reference)} {error or ''}", flush=True)
                save()
        result["summary"] = {}
        for group in ("all", "clean", "other", "short", "silence"):
            rows = [r for r in result["rows"] if (group == "all" and r["group"] != "silence") or
                    (group == "short" and r["group"] != "silence" and r["audio_seconds"] <= 10) or r["group"] == group]
            if not rows:
                continue
            count = sum(r["reference_words"] for r in rows)
            times = sorted(r["elapsed_seconds"] for r in rows)
            result["summary"][group] = dict(clips=len(rows), reference_words=count,
                wer=sum(r["edits"] for r in rows)/count if count else None,
                median_seconds=statistics.median(times), p95_seconds=times[max(0, math.ceil(len(times)*0.95)-1)],
                mean_rtf=sum(times)/sum(r["audio_seconds"] for r in rows),
                failures=sum(r["error"] is not None for r in rows),
                nonempty=sum(bool(r["hypothesis"]) for r in rows))
        if args.mode == "resident":
            result["end_memory_kib"] = worker_memory(backend)
        else:
            result["child_maxrss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        result["load_end"] = Path("/proc/loadavg").read_text().strip()
        result["finished"] = datetime.now(timezone.utc).isoformat()
        save()
    finally:
        if args.mode == "resident" and backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
