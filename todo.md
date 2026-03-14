# ptt-realtime branch TODO

## Done
- [x] kglobal_hotkey.py: removed `NO_TRANSCRIPT_SENTINEL` dead code
- [x] Extract shared `load_whisper_model()` into `whisper_common.py` (was duplicated across dictate.py, transcribe.py, benchmark.py, eval/sweep.py, eval/evaluate.py, mic_realtime.py)
- [x] Extract shared `transcribe_pcm()` into `whisper_common.py` (was duplicated in dictate.py and mic_realtime.py)
- [x] Extract shared `VADSegmenter` class into `whisper_common.py` (was duplicated in dictate.py and mic_realtime.py)
- [x] Fix `_transcribe_pcm` hardcoding `condition_on_previous_text=False` and `vad_filter=False` instead of passing through CLI args (dictate.py)
- [x] Fix race condition in `DictationDaemon.start_recording` — threads now created inside lock
- [x] Add `timeout` to all `thread.join()` calls in mic_realtime.py
- [x] Document lock semantics in `DictationDaemon.__init__`
- [x] `eval/evaluate.py:73` — removed unreachable return referencing undefined `samples`
- [x] `eval/evaluate.py` — removed unused imports: `soundfile`, `numpy`, `hf_hub_download`

## Not yet started

### Medium: Documentation
- [ ] Update `README.md`: document new streaming args (`--block-ms`, `--energy-threshold`, `--silence-ms`, `--min-speech-ms`, `--start-speech-ms`, `--max-utterance-s`)
- [ ] Update `README.md`: add install/uninstall instructions, service file location (`~/.config/systemd/user/`)
- [ ] Update `README.md`: note that daemon now streams transcription in real-time during PTT
