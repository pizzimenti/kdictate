"""Check native readiness and cleanup on an isolated D-Bus session.

Run under dbus-run-session. This opens no microphone and never uses the desktop
session bus. Runtime files and logs go under the given output directory.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=('whisper','parakeet'), required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('DBUS_SESSION_BUS_ADDRESS', '').startswith('unix:path=/tmp/dbus-'):
        parser.error('run this check under dbus-run-session, away from the desktop bus')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['XDG_STATE_HOME'] = str(args.output_dir/'logs')
    os.environ['KDICTATE_STT_WORKER'] = str(args.worker.resolve())
    from kdictate import backend
    from kdictate.core import daemon
    # Test-only model path injection avoids provisioning a second production
    # model merely to exercise the actual daemon's selected-engine path.
    setattr(backend, 'PARAKEET_MODEL_PATH' if args.engine=='parakeet' else 'GGML_MODEL_PATH', args.model.resolve())
    create = backend.create_gpu_backend
    recorded = {}
    timers = []
    configure = daemon.configure_logging
    class Ready(logging.Handler):
        def emit(self, record):
            if record.getMessage() == 'daemon started':
                recorded['daemon_ready'] = True
                timer = threading.Timer(.25, lambda: os.kill(os.getpid(), signal.SIGTERM))
                timer.daemon = True
                timer.start()
                timers.append(timer)
    def observe_logging(*a, **kw):
        logger = configure(*a, **kw)
        logger.addHandler(Ready())
        return logger
    daemon.configure_logging = observe_logging
    def observe(config):
        result = create(config)
        if result is None:
            return None
        recorded.update(engine=result.engine, worker_pid=result._process.pid,
                        load_seconds=result.startup_seconds)
        return result
    backend.create_gpu_backend = observe
    result = daemon.main(['--backend', 'parakeet' if args.engine=='parakeet' else 'gpu',
                          '--state-file', str(args.output_dir/'state'),
                          '--last-text-file', str(args.output_dir/'last-text')])
    for timer in timers:
        timer.cancel()
    assert result == 0 and recorded.get('daemon_ready'), f'daemon failed to initialize: {result}'
    try:
        os.kill(recorded['worker_pid'], 0)
    except ProcessLookupError:
        recorded['worker_reaped'] = True
    else:
        raise AssertionError('native worker survived daemon shutdown')
    recorded['exit_code'] = result
    (args.output_dir/'result.json').write_text(json.dumps(recorded, indent=2)+'\n')
    print(json.dumps(recorded))


if __name__=='__main__':
    main()
