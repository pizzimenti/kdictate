"""Counterbalance CLI timings and check resident decode after idle periods."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.run import pcm, words, edit_distance, process_memory
from kdictate.backend import WhisperCppBackend
from kdictate.resident_backend import ResidentBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--old-cli', type=Path, required=True)
    parser.add_argument('--new-bin-dir', type=Path, required=True)
    parser.add_argument('--whisper-model', type=Path, required=True)
    parser.add_argument('--parakeet-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    samples = json.loads(args.manifest.read_text())['samples']
    for sample in samples:
        assert hashlib.sha256(Path(sample['path']).read_bytes()).hexdigest() == sample['sha256']
    short = [s for s in samples if s['reference'] and s['seconds']<=10]
    chosen = [short[round(i*(len(short)-1)/7)] for i in range(8)]
    adapters = {
        'old': WhisperCppBackend(str(args.old_cli), args.whisper_model),
        'new': WhisperCppBackend(str(args.new_bin_dir/'whisper-cli'), args.whisper_model),
    }
    result = dict(cli_pairs=[], idle={}, manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  load_start=Path('/proc/loadavg').read_text().strip())
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    for i, sample in enumerate(chosen):
        audio=pcm(sample['path'])
        row=dict(id=sample['id'], audio_seconds=sample['seconds'], reference=sample['reference'])
        order=('old','new') if i%2==0 else ('new','old')
        row['order']=order
        for key in order:
            start=time.perf_counter()
            text=adapters[key].transcribe([audio],sample['seconds'])
            row[key]=dict(seconds=time.perf_counter()-start,hypothesis=text,
                          edits=edit_distance(words(sample['reference']),words(text)))
        result['cli_pairs'].append(row)
        print(sample['id'],row['old']['seconds'],row['new']['seconds'],flush=True)
        save()
    result['cli_median_new_over_old'] = statistics.median(r['new']['seconds']/r['old']['seconds'] for r in result['cli_pairs'])
    sample=short[0]
    audio=pcm(sample['path'])
    for engine,model in [('whisper',args.whisper_model),('parakeet',args.parakeet_model)]:
        backend=ResidentBackend(str(args.new_bin_dir/'kdictate-stt-worker'),model,engine=engine)
        try:
            backend.transcribe([np.zeros(16000,dtype=np.int16)],1)
            # Exercise the largest input before measuring retained buffers.
            longest=max((s for s in samples if s['reference']),key=lambda s:s['seconds'])
            backend.transcribe([pcm(longest['path'])],longest['seconds'])
            rows=[]
            result['idle'][engine]=dict(id=sample['id'],reference=sample['reference'],audio_seconds=sample['seconds'],rows=rows)
            for delay in (0,10,30):
                print(f'{engine}: waiting {delay}s before decode',flush=True)
                time.sleep(delay)
                start=time.perf_counter()
                text=backend.transcribe([audio],sample['seconds'])
                elapsed=time.perf_counter()-start
                rows.append(dict(idle_seconds=delay,elapsed_seconds=elapsed,hypothesis=text,
                                 memory_kib=process_memory(backend._process.pid)))
                print(engine,delay,elapsed,text,flush=True)
                save()
        finally:
            backend.close()
    result['load_end']=Path('/proc/loadavg').read_text().strip()
    save()


if __name__=='__main__':
    main()
