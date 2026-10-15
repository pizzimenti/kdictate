"""Exercise model selection, verification, and service configuration."""
from dataclasses import replace
import hashlib
from pathlib import Path
from unittest import mock

import pytest

import install
from kdictate import backend
from kdictate.config import DictationConfig, parse_args
from kdictate.exceptions import TranscriptionError


@pytest.fixture
def ctx(tmp_path):
    return install.InstallContext(script_path=tmp_path/'install.py', script_dir=tmp_path,
                                  home=tmp_path, runtime_dir=tmp_path/'runtime', backend='parakeet')


@pytest.mark.parametrize('choice,expected', [('', 'gpu'), ('1', 'gpu'), ('2', 'cpu'), ('3', 'parakeet')])
def test_installer_selects_one_backend(choice, expected):
    with mock.patch.object(install, '_detect_gpu', return_value=('/test/worker', [])), mock.patch('builtins.input', return_value=choice):
        assert install._prompt_backend() == expected


def test_unavailable_gpu_selects_cpu():
    with mock.patch.object(install, '_detect_gpu', return_value=(None, ['no Vulkan GPU'])), mock.patch('builtins.input', return_value='y'):
        assert install._prompt_backend() == 'cpu'


@pytest.mark.parametrize('selected', ['gpu', 'parakeet', 'cpu'])
def test_service_and_templates_keep_explicit_choice(ctx, selected):
    ctx = replace(ctx, backend=selected)
    install._write_backend_dropin(ctx)
    text = (ctx.home/'.config/systemd/user'/f'{install.SERVICE_NAME}.d/10-backend.conf').read_text()
    assert f'--backend {selected}' in text
    assert ctx.replacements['@@BACKEND_FLAGS@@'] == f' --backend {selected}'
    assert ('--model-dir' in text) == (selected == 'cpu')


@pytest.mark.parametrize('selected', ['gpu', 'parakeet'])
def test_model_download_is_pinned_and_checksum_verified(ctx, selected):
    ctx = replace(ctx, backend=selected)
    payload = b'test verified model'
    expected = hashlib.sha256(payload).hexdigest()
    prefix = 'PARAKEET' if selected == 'parakeet' else 'GGML'
    path = ctx.runtime_dir / getattr(install, f'{prefix}_MODEL_FILENAME')
    def download(command, **kwargs):
        assert repr(getattr(install, f'{prefix}_MODEL_HF_REPO')) in command[-1]
        assert repr(getattr(install, f'{prefix}_MODEL_REVISION')) in command[-1]
        assert 'force_download=True' in command[-1]
        path.write_bytes(payload)
    with mock.patch.object(install, f'{prefix}_MODEL_SHA256', expected), mock.patch.object(install, '_hf_download_env', return_value=('python', None)), mock.patch.object(install.subprocess, 'run', side_effect=download) as run:
        # Truncated or wrong existing data must not satisfy the offline check.
        path.parent.mkdir(parents=True)
        path.write_bytes(b'broken')
        install.download_gpu_model(ctx)
        assert run.call_count == 1
        install.download_gpu_model(ctx)
        assert run.call_count == 1


def test_corrupt_model_download_aborts(ctx):
    with mock.patch.object(install, '_hf_download_env', return_value=('python', None)), mock.patch.object(install.subprocess, 'run'):
        with pytest.raises(SystemExit):
            install.download_gpu_model(ctx)


@pytest.mark.parametrize('selected,engine,constant', [('gpu','whisper','GGML_MODEL_PATH'), ('parakeet','parakeet','PARAKEET_MODEL_PATH')])
def test_gpu_runtime_loads_and_warms_only_selected_model(tmp_path, selected, engine, constant):
    config = DictationConfig.from_namespace(parse_args(['--backend', selected]))
    model = tmp_path/'model.bin'
    model.touch()
    with mock.patch.object(backend, 'find_stt_worker', return_value='/worker'), mock.patch.object(backend, constant, model), mock.patch.object(backend, 'ResidentBackend') as resident:
        result = backend.create_gpu_backend(config)
        assert result is resident.return_value
        assert resident.call_args.args == ('/worker', model)
        assert resident.call_args.kwargs['engine'] == engine
        assert resident.call_args.kwargs['beam_size'] == config.beam_size
        resident.return_value.transcribe.assert_called_once()


def test_missing_selected_model_does_not_fall_back(tmp_path):
    config = DictationConfig.from_namespace(parse_args(['--backend', 'parakeet']))
    with mock.patch.object(backend, 'find_stt_worker', return_value='/worker'), mock.patch.object(backend, 'PARAKEET_MODEL_PATH', tmp_path/'missing'), mock.patch.object(backend, 'ResidentBackend') as resident:
        assert backend.create_gpu_backend(config) is None
        resident.assert_not_called()


def test_warmup_failure_closes_worker(tmp_path):
    model = tmp_path/'model.bin'
    model.touch()
    config = DictationConfig.from_namespace(parse_args(['--backend', 'parakeet']))
    with mock.patch.object(backend, 'find_stt_worker', return_value='/worker'), mock.patch.object(backend, 'PARAKEET_MODEL_PATH', model), mock.patch.object(backend, 'ResidentBackend') as resident:
        resident.return_value.transcribe.side_effect = TranscriptionError('GPU failure')
        assert backend.create_gpu_backend(config) is None
        resident.return_value.close.assert_called_once()


@pytest.mark.parametrize('selected,explicit,expected', [
    ('cpu', None, 1), ('gpu', None, 3), ('gpu', 1, 1), ('gpu', 5, 5), ('cpu', 3, 3),
])
def test_backend_beam_defaults_and_explicit_overrides(selected, explicit, expected):
    argv = ['--backend', selected]
    if explicit is not None:
        argv += ['--beam-size', str(explicit)]
    assert DictationConfig.from_namespace(parse_args(argv)).beam_size == expected


def test_gpu_factory_forwards_explicit_beam_size(tmp_path):
    model = tmp_path/'model.bin'
    model.touch()
    config = DictationConfig.from_namespace(parse_args(['--backend', 'gpu', '--beam-size', '5']))
    with mock.patch.object(backend, 'find_stt_worker', return_value='/worker'), mock.patch.object(backend, 'GGML_MODEL_PATH', model), mock.patch.object(backend, 'ResidentBackend') as resident:
        assert backend.create_gpu_backend(config) is resident.return_value
        assert resident.call_args.kwargs['beam_size'] == 5
