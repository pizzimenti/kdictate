"""Check portable inputs and complete failure reports through the real harness."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import wave
from unittest import mock

import numpy as np
import pytest

from benchmarks import report, run


RESULTS = Path(__file__).resolve().parents[2]/'benchmarks/results/2026-10-15'


def test_last_worker_failure_still_saves_completed_results(tmp_path):
    worker = tmp_path/'worker'
    worker.write_text('''#!/usr/bin/python
import struct, sys
ready = ("READY 1 " + sys.argv[1]).encode()
sys.stdout.buffer.write(struct.pack("<I", len(ready)) + ready)
sys.stdout.buffer.flush()
sys.stdin.buffer.read(4)
sys.exit(3)
''')
    worker.chmod(0o755)
    model = tmp_path/'model.bin'
    model.write_bytes(b'test model')
    clip = tmp_path/'clip.wav'
    with wave.open(str(clip), 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        audio.writeframes(np.zeros(160, dtype='<i2').tobytes())
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(dataset='test', revision='test', samples=[dict(
        id='clip', group='clean', path='clip.wav', reference='hello', seconds=.01,
        sha256=hashlib.sha256(clip.read_bytes()).hexdigest())])))
    output = tmp_path/'result.json'
    argv = ['benchmark', '--manifest', str(manifest), '--binary', str(worker),
            '--model', str(model), '--mode', 'resident', '--no-warmup', '--name', 'crash',
            '--output', str(output)]
    with mock.patch.object(sys, 'argv', argv), mock.patch.object(run, 'vulkan_summary', return_value='test'):
        run.main()
    result = json.loads(output.read_text())
    assert result['rows'][0]['error']
    assert result['summary']['all']['failures'] == 1
    assert result['summary']['all']['wer'] == 1
    assert result['end_memory_kib'] is None
    assert result['finished']


@pytest.mark.parametrize('matching', [True, False])
def test_report_only_includes_idle_checks_for_its_audio_manifest(tmp_path, matching):
    data = json.loads((RESULTS/'whisper-1.9.5-resident.json').read_text())
    check = json.loads((RESULTS/'responsiveness.json').read_text())
    check['manifest_sha256'] = data['manifest_sha256'] if matching else 'another dataset'
    (tmp_path/'responsiveness.json').write_text(json.dumps(check))
    source = tmp_path/'result.json'
    source.write_text(json.dumps(data))
    output = tmp_path/'report.html'
    report.report([source], output)
    assert ('Idle response and timing cross-check' in output.read_text()) == matching


def test_report_handles_missing_memory_after_failed_decode(tmp_path):
    data = json.loads((RESULTS/'whisper-1.9.5-resident.json').read_text())
    data['idle_memory_kib'] = None
    data['end_memory_kib'] = None
    source = tmp_path/'result.json'
    source.write_text(json.dumps(data))
    output = tmp_path/'report.html'
    report.report([source], output)
    assert 'unavailable' in output.read_text()


def test_report_rejects_comparisons_from_different_manifests(tmp_path):
    first = json.loads((RESULTS/'whisper-1.9.5-resident.json').read_text())
    second = copy.deepcopy(first)
    second['manifest_sha256'] = 'another dataset'
    paths = [tmp_path/'first.json', tmp_path/'second.json']
    for path, data in zip(paths, (first, second)):
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='different audio manifests'):
        report.report(paths, tmp_path/'report.html')
