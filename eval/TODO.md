# whisper-dictate evaluation TODO

## Baseline results (beam=5, no VAD)
- Avg WER: 7.2%
- Speed: 1.8x real-time (RTF 0.541)
- Notable: short clips (<4s) have RTF >1.0 (slower than real-time) due to fixed overhead

## Remaining benchmark runs
- [ ] beam=1, no VAD — measure speed gain vs accuracy loss
- [ ] beam=5, VAD on — should fix hallucination on silence
- [ ] beam=1, VAD on — best speed config
- [ ] beam=5, VAD on, condition_on_previous=True — check if coherence improves

## Accuracy fixes to apply to dictate.py
- [x] Add `vad_filter=True` — critical, strips silence, prevents hallucination garbage
- [x] Add `condition_on_previous_text=False` — prevents cascading hallucinations
- [x] Add `no_speech_threshold=0.6` — reject low-confidence segments
- [ ] Pick beam_size based on benchmark results (1 vs 5 tradeoff)

## Speed fixes to investigate
- [ ] Reduce beam_size (1 is ~2-3x faster, slight accuracy cost)
- [ ] ~4s minimum decode time even for short clips — investigate overhead
- [ ] Test with fewer cpu_threads (6 vs 12) to check if contention hurts

## UX improvements
- [ ] Persistent notification while recording (replace-mode with fixed ID)
- [ ] Fix stale venv shebangs from whisper-cli rename (recreate venv or sed fix)
- [ ] Commit all pending changes once tuning is done

## Open design questions
- How should dictation sessions work? (continuous vs toggle vs push-to-talk)
- How to clearly indicate recording state to the user?
