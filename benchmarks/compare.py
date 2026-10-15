"""Run the five speech configurations serially, then render their report."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--old-cli', type=Path, required=True)
    parser.add_argument('--new-bin-dir', type=Path, required=True)
    parser.add_argument('--whisper-model', type=Path, required=True)
    parser.add_argument('--parakeet-model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    configs = [
        ('whisper-1.8.6-cli', 'Whisper 1.8.6 CLI (current)', 'whisper', 'cli', args.old_cli, args.whisper_model),
        ('whisper-1.9.5-cli', 'Whisper 1.9.5 CLI', 'whisper', 'cli', args.new_bin_dir/'whisper-cli', args.whisper_model),
        ('whisper-1.9.5-resident', 'Whisper 1.9.5 resident', 'whisper', 'resident', args.new_bin_dir/'kdictate-stt-worker', args.whisper_model),
        ('parakeet-cli', 'Parakeet v3 Q8 CLI', 'parakeet', 'cli', args.new_bin_dir/'parakeet-cli', args.parakeet_model),
        ('parakeet-resident', 'Parakeet v3 Q8 resident', 'parakeet', 'resident', args.new_bin_dir/'kdictate-stt-worker', args.parakeet_model),
    ]
    root = Path(__file__).resolve().parent
    outputs = []
    for key, name, engine, mode, binary, model in configs:
        output = args.output_dir/f'{key}.json'
        subprocess.run([sys.executable, str(root/'run.py'), '--manifest', str(args.manifest),
            '--binary', str(binary.resolve()), '--model', str(model.resolve()), '--engine', engine,
            '--mode', mode, '--name', name, '--output', str(output)], check=True)
        outputs.append(str(output))
    subprocess.run([sys.executable, str(root/'report.py'), '--output', str(args.output_dir/'report.html'), *outputs], check=True)


if __name__ == '__main__':
    main()
