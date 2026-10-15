"""Generate a standalone HTML comparison from the recorded benchmark JSON."""
from __future__ import annotations

import argparse
import csv
from html import escape as h
import json
from pathlib import Path
import random
import statistics


def percent(value):
    return '—' if value is None else f'{value*100:.2f}%'


def paired_interval(left, right):
    """Paired speaker-cluster bootstrap; descriptive for this small corpus."""
    a = {(r['id'],r['repeat']): r for r in left['rows'] if r['reference_words']}
    b = {(r['id'],r['repeat']): r for r in right['rows'] if r['reference_words']}
    if a.keys() != b.keys():
        return None
    speakers = {}
    for key, row in a.items():
        speaker = (row['group'], row['speaker'])
        words, delta = speakers.get(speaker, (0, 0))
        speakers[speaker] = words+row['reference_words'], delta+b[key]['edits']-row['edits']
    clusters = list(speakers.values())
    rng = random.Random(20261015)
    deltas = []
    for _ in range(3000):
        sample = rng.choices(clusters, k=len(clusters))
        deltas.append(sum(x[1] for x in sample)/sum(x[0] for x in sample)*100)
    deltas.sort()
    return deltas[75], deltas[2924]


def report(paths, output):
    results = [json.loads(p.read_text()) for p in paths]
    if len({r['manifest_sha256'] for r in results}) != 1:
        raise ValueError('Cannot compare results from different audio manifests')
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KDictate 1.0 speech comparison</title><style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 20px;color:#183040;background:#fafcfd}
h1,h2{line-height:1.2} table{border-collapse:collapse;width:100%;margin:20px 0;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:10px;border-bottom:1px solid #cbd5df;vertical-align:top} th{background:#e9f1f6}
.scroll{overflow-x:auto} .note{background:#edf3f6;padding:16px;border-radius:8px} details{margin:12px 0;border:1px solid #ccd7df;border-radius:6px;padding:12px}
summary{cursor:pointer;font-weight:600} code{overflow-wrap:anywhere;font-size:13px} input{font:inherit;padding:10px;width:min(95%,600px)}
.bad{color:#a62d23} .good{color:#17633f} footer{font-size:14px;color:#4a5d69} a{color:#17609a}
</style><h1>KDictate 1.0 release candidate: speech comparison</h1>''']
    speech = [r for r in results[0]['rows'] if r['reference_words']]
    speakers = len({(r['group'], r['speaker']) for r in speech})
    parts.append(f'<p>{len(speech)} human speech clips, {speakers} speakers, {sum(r["audio_seconds"] for r in speech)/60:.1f} minutes, '
                 f'{sum(r["reference_words"] for r in speech):,} reference words; two additional digital-silence checks. '
                 f'Host: {h(results[0]["cpu"])}.</p><pre>{h(results[0]["vulkan"])}</pre>')
    parts.append('<p class="note">Lower WER and lower latency are better. CLI timing includes a fresh process and model load for each clip. '
                 'Resident timing follows startup and one second of warmup silence. These measurements exclude microphone capture, '
                 'silence detection and desktop insertion. OS file caches are warm; the installed dictation service was left unchanged.</p>')
    parts.append('<p>Measurements were taken in an active desktop session with other processes running. '
                 'Run-start and run-end load snapshots are retained in each raw JSON file.</p>')
    parts.append('<h2>Accuracy and response time</h2><div class="scroll"><table><tr><th>Configuration</th><th>WER, clean</th><th>WER, harder</th>'
                 '<th>WER, all</th><th>Median decode</th><th>p95 decode</th><th>≤10s clips median</th><th>Real-time factor</th><th>Failures</th></tr>')
    csv_rows = []
    for r in results:
        s=r['summary']; a=s['all']
        parts.append(f'<tr><td>{h(r["name"])}</td><td>{percent(s["clean"]["wer"])}</td><td>{percent(s["other"]["wer"])}</td>'
            f'<td>{percent(a["wer"])}</td><td>{a["median_seconds"]:.3f}s</td><td>{a["p95_seconds"]:.3f}s</td>'
            f'<td>{s["short"]["median_seconds"]:.3f}s</td><td>{a["mean_rtf"]:.3f}</td><td>{a["failures"]}</td></tr>')
        csv_rows.append(dict(name=r['name'], **a, clean_wer=s['clean']['wer'], other_wer=s['other']['wer']))
    parts.append('</table></div><p>WER is total word edits / total reference words; lowercase and punctuation normalization only. '
                 'Numbers and contractions remain literal. Failed requests count as empty hypotheses. A real-time factor of 0.1 means '
                 'one second of processing per ten seconds of audio.</p>')
    by_engine={r['engine']:r for r in results if r['mode']=='resident'}
    if 'whisper' in by_engine and 'parakeet' in by_engine:
        whisper=by_engine['whisper']; parakeet=by_engine['parakeet']
        delta=sum(r['edits'] for r in parakeet['rows'])-sum(r['edits'] for r in whisper['rows'])
        comparison='the same number of word errors as' if delta==0 else f'{abs(delta)} {"more" if delta>0 else "fewer"} word errors than'
        parts.append(f'<p class="note">Parakeet is a viable option on this host: median resident decode '
            f'{parakeet["summary"]["all"]["median_seconds"]:.3f}s, versus '
            f'{whisper["summary"]["all"]["median_seconds"]:.3f}s for resident Whisper. '
            f'It made {comparison} updated Whisper in this subset. '
            'Whisper remains the installer default; Parakeet is offered as an optional choice pending validation with your microphone.</p>')
    for engine in ('whisper','parakeet'):
        modes={r['mode']:r for r in results if r['engine']==engine}
        if 'cli' in modes and 'resident' in modes:
            a=modes['cli']['rows']; b=modes['resident']['rows']
            if len(a)==len(b) and all((x['id'],x['hypothesis'])==(y['id'],y['hypothesis']) for x,y in zip(a,b)):
                parts.append(f'<p>{h(engine.title())}: CLI and resident transcripts matched exactly on all {len(a)} clips.</p>')
    parts.append('<h2>Startup and retained memory</h2><table><tr><th>Configuration</th><th>Model load</th><th>Warmup</th><th>Idle RSS</th><th>Peak RSS</th><th>Silence with text</th></tr>')
    for r in results:
        resident = r['mode']=='resident'
        idle_memory = r.get('idle_memory_kib') or {}
        end_memory = r.get('end_memory_kib') or {}
        idle = (f'{idle_memory["VmRSS"]/1024:.0f} MiB' if 'VmRSS' in idle_memory else 'unavailable') if resident else 'process exits'
        peak = end_memory.get('VmHWM') if resident else r.get('child_maxrss_kib')
        peak_text = f'{peak/1024:.0f} MiB' if peak is not None else 'unavailable'
        startup = f'{r["startup_seconds"]:.3f}s' if resident else 'in each decode'
        warmup = f'{r.get("warmup_seconds",0):.3f}s' if resident else 'in each decode'
        parts.append(f'<tr><td>{h(r["name"])}</td><td>{startup}</td><td>{warmup}</td><td>{idle}</td>'
                     f'<td>{peak_text}</td><td>{r["summary"]["silence"]["nonempty"]}/2</td></tr>')
    parts.append('</table><p>RSS is Linux process memory, not a complete GPU allocation measure. Shared memory and device memory accounting vary. '
                 'A resident model keeps these allocations while idle until the daemon stops.</p>')
    followup = output.parent/'responsiveness.json'
    if followup.exists():
        check = json.loads(followup.read_text())
        if check.get('load_end') and check.get('manifest_sha256') == results[0]['manifest_sha256']:
            ratio=check['cli_median_new_over_old']
            parts.append('<h2>Idle response and timing cross-check</h2>'
                f'<p>Eight short clips were tested again with old/new CLI order alternating between clips. '
                f'The median per-clip new/old timing ratio was {ratio:.3f} '
                '(1.0 means equal speed). This cross-check helps separate engine changes from changing desktop load.</p>'
                '<table><tr><th>Resident engine</th><th>Immediately</th><th>After 10s idle</th><th>After 30s idle</th><th>GPU allocations retained</th></tr>')
            for engine, data in check['idle'].items():
                rows=data['rows']; memory=rows[-1]['memory_kib']
                total=sum(memory.get(k,0) for k in ('drm-total-vram','drm-total-gtt'))
                allocation=f'{total/1024:.0f} MiB' if total else 'not exposed'
                parts.append(f'<tr><td>{h(engine)}</td>' + ''.join(f'<td>{r["elapsed_seconds"]:.3f}s</td>' for r in rows) + f'<td>{allocation}</td></tr>')
            parts.append('</table><p>The idle test repeats one short clip after warmup and the corpus\'s longest clip. '
                'GPU figures come from per-client Linux DRM VRAM + GTT allocation counters, with duplicate file descriptors excluded. '
                'On this integrated GPU these allocations consume shared system memory; RSS should not be added blindly because driver mappings can overlap. '
                '<a href="responsiveness.json">Full timing cross-check, idle transcripts and raw memory counters</a>.</p>')
    if len(results)>1:
        parts.append('<h2>Accuracy uncertainty</h2><p>Differences against the first configuration, in percentage points of WER '
                     '(negative means fewer errors). The interval is a paired, speaker-cluster bootstrap (3,000 draws). '
                     'This small audiobook subset cannot establish accuracy for your microphone or all dictation tasks.</p><ul>')
        baseline=results[0]['summary']['all']['wer']
        for r in results[1:]:
            interval=paired_interval(results[0],r)
            if interval:
                parts.append(f'<li>{h(r["name"])}: {(r["summary"]["all"]["wer"]-baseline)*100:+.2f} points; '
                             f'95% interval [{interval[0]:+.2f}, {interval[1]:+.2f}].</li>')
        parts.append('</ul>')
    parts.append('<h2>Inspect every transcript</h2><p>Search by clip ID, reference text or recognized text.</p>'
                 '<input id="search" type="search" placeholder="Filter clips…" aria-label="Filter clips"><div id="clips">')
    for i, row in enumerate(results[0]['rows']):
        parts.append(f'<details><summary>{h(row["id"])} · {row["audio_seconds"]:.2f}s</summary><p><b>Reference:</b> {h(row["reference"]) or "(silence)"}</p><table>')
        for r in results:
            other = r['rows'][i]
            assert (other['id'],other['reference'])==(row['id'],row['reference'])
            parts.append(f'<tr><td>{h(r["name"])}</td><td>{other["elapsed_seconds"]:.3f}s</td><td>{other["edits"]} edits</td>'
                         f'<td>{h(other["hypothesis"]) or "(empty)"}{" — ERROR: "+h(other["error"]) if other["error"] else ""}</td></tr>')
        parts.append('</table></details>')
    parts.append('</div><h2>Reproducibility</h2><ul>')
    for p,r in zip(paths,results):
        parts.append(f'<li><a href="{h(p.name)}">{h(r["name"])} raw JSON</a>; run {h(r["started"])}; threads {r["threads"]}, '
                     f'beam {r["beam"]}; model SHA-256 <code>{r["model_sha256"]}</code>; '
                     f'binary SHA-256 <code>{r["binary_sha256"]}</code>.</li>')
    parts.append(f'</ul><p>Manifest SHA-256: <code>{results[0]["manifest_sha256"]}</code>. '
                 f'Pinned dataset revision: <code>{results[0]["dataset_revision"]}</code>. '
                 'Uniform row offsets from test-clean and test-other; no accuracy-based filtering. All configurations run serially. '
                 'Retained JSON includes every hypothesis, timing, error, host load and memory snapshot.</p>')
    if any(r.get('original_manifest_sha256') for r in results):
        parts.append('<p>Recorded audio paths were normalized to manifest-relative filenames for portability. '
            'Original manifest hashes are retained in the raw JSON; audio checksums, references, hypotheses and timings are unchanged. '
            'Generate the audio and its manifest with prepare_audio.py before rerunning the benchmark.</p>')
    parts.append('<footer>LibriSpeech: Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur (2015); recordings from LibriVox. '
                 '<a href="https://www.openslr.org/12">Dataset and attribution</a>, '
                 '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. '
                 '<a href="https://huggingface.co/datasets/openslr/librispeech_asr">Pinned Parquet mirror</a>. '
                 'Limitations: read English audiobooks, small sample, no live end-to-end desktop test in this report.</footer>'
                 '<script>document.querySelector("#search").oninput=e=>{const q=e.target.value.toLowerCase();'
                 'document.querySelectorAll("#clips details").forEach(d=>d.hidden=!d.textContent.toLowerCase().includes(q))}</script></html>')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(parts))
    with output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(csv_rows)
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('results',nargs='+',type=Path)
    args=parser.parse_args()
    report(args.results,args.output)
