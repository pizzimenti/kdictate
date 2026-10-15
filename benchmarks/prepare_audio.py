"""Prepare a pinned, speaker-diverse LibriSpeech subset with gold transcripts.

Run with requirements-benchmarks.txt installed. Large downloads stay on disk.
The Parquet mirror avoids thousands of dataset-viewer/CDN requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave

SOURCE = "https://huggingface.co/datasets/openslr/librispeech_asr"
REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
CHECKSUMS = {
    "clean": "7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0",
    "other": "38e0c86a8104585c577badd707ca4331e20fa2c645f46180af0b7bcdecff9249",
}


def prepare(root: Path, per_split: int, parquet_root: Path) -> Path:
    import pyarrow.parquet as pq
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("clean", "other"):
        filename = f"{split}/test/0000.parquet"
        archive = parquet_root / filename
        if not archive.exists():
            from huggingface_hub import hf_hub_download
            archive = Path(hf_hub_download(repo_id="openslr/librispeech_asr", repo_type="dataset",
                           revision=REVISION, filename=filename, local_dir=str(parquet_root)))
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKSUMS[split]:
                raise ValueError(f"dataset checksum mismatch: {archive}")
        table = pq.read_table(archive)
        if per_split > table.num_rows:
            raise ValueError("sample count exceeds split length")
        indices = [round(i*(table.num_rows-1)/(per_split-1)) for i in range(per_split)]
        for record in table.take(indices).to_pylist():
            clip_id = record["id"]
            name = f"{split}-{clip_id}"
            output = root / f"{name}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "pipe:0", "-ar", "16000",
                            "-ac", "1", "-c:a", "pcm_s16le", str(output)],
                           input=record["audio"]["bytes"], check=True)
            with wave.open(str(output)) as audio:
                duration = audio.getnframes()/audio.getframerate()
            rows.append(dict(id=name, group=split, path=output.name, reference=record["text"],
                             speaker=int(record["speaker_id"]), seconds=duration,
                             sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                             source=f"{SOURCE}/blob/{REVISION}/{filename}"))
        del table
    # WER is undefined for an empty reference. Silence checks are separate.
    for seconds in (1, 5):
        path = root / f"silence-{seconds}.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(seconds * 16000 * 2))
        rows.append(dict(id=path.stem, group="silence", path=path.name, reference="", seconds=seconds,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source="generated digital silence"))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(dict(dataset=SOURCE, revision=REVISION, license="CC-BY-4.0",
        attribution="LibriSpeech: Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur (2015); LibriVox recordings",
        selection="uniform row offsets in each pinned test Parquet split; no accuracy-based filtering",
        samples=rows), indent=2) + "\n")
    print(f"{len(rows)} clips, {sum(r['seconds'] for r in rows):.1f}s; {manifest}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cache = Path.home()/".cache/kdictate-benchmarks"
    parser.add_argument("--directory", type=Path, default=cache/"audio")
    parser.add_argument("--parquet-root", type=Path, default=cache/"parquet")
    parser.add_argument("--per-split", type=int, default=30)
    args = parser.parse_args()
    if args.per_split < 2:
        parser.error("--per-split must be at least 2")
    prepare(args.directory, args.per_split, args.parquet_root)
