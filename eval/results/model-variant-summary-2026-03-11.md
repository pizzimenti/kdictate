# Whisper Variant Summary - 2026-03-11

Historical note: this document is retained as evaluation data. The active project runtime was later standardized on `distil-medium-en`.

## Hardware and runtime

- Host CPU: AMD Ryzen 5 8640HS, 6 physical cores / 12 logical threads
- GPU present: AMD Radeon 760M
- Runtime actually used: CPU-only
- Why CPU-only: the installed `ctranslate2` reported `cuda_available=0`, `torch` is a CPU build in this repo, and `whisper-dictate` currently resolves runtime to CPU only
- Eval set: bundled 20-sample LibriSpeech manifest at `eval/audio/manifest.json`
- Accuracy metric: normalized WER from `eval/sweep.py`, which strips case/punctuation/hyphenation noise before token comparison
- Decode settings for the comparison runs below:
  - `beam_size=1`
  - `best_of=1`
  - `temperature=0.0`
  - `condition_on_previous_text=False`
  - `without_timestamps=True`
  - `vad_filter=False`

## Result files

- Previous baseline matrix: `eval/results/sweeps/20260311_100929_initial/`
- Thread follow-up: `eval/results/sweeps/20260311_102448_threads_followup/`
- New variant sweep: `eval/results/sweeps/20260311_114807_new_variants/`

## Headline results

| Model | Threads | Avg WER | Overall RTF | Short clip mean s | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `whisper-large-v3-turbo` | 12 | 1.61% | 0.589 | 4.697 | Best accuracy from tested configs |
| `distil-large-v3` | 6 | 1.83% | 0.592 | 4.856 | Better than `distil-large-v3.5` on this machine, still slow |
| `distil-medium-en` | 6 | 2.49% | 0.361 | 2.914 | Best balanced default |
| `distil-large-v3.5-ct2` | 6 | 2.75% | 0.600 | 4.870 | Not competitive here |
| `whisper-small.en` | 6 | 2.81% | 0.165 | 1.201 | Best speed / still usable |
| `whisper-base.en` | 12 | 4.36% | 0.074 | 0.516 | Fastest by far, but accuracy loss is noticeable |

## New model detail

### `distil-large-v3.5-ct2`

- Downloaded size: about 1.5 GB
- `t12`: avg WER `2.75%`, RTF `0.606`, short clips `4.894s`
- `t6`: avg WER `2.75%`, RTF `0.600`, short clips `4.870s`
- Conclusion: not worth keeping on this machine. It was slower than `distil-medium-en`, less accurate than `turbo`, and not better than the older `distil-large-v3` baseline we already had.

### `whisper-small.en`

- Downloaded size: about 462 MB
- `t12`: avg WER `2.81%`, RTF `0.182`, short clips `1.292s`
- `t6`: avg WER `2.81%`, RTF `0.165`, short clips `1.201s`
- Conclusion: strongest new result. Accuracy is a bit worse than `distil-medium-en`, but latency is dramatically better. This is the best "fast mode" candidate.

### `whisper-base.en`

- Downloaded size: about 139 MB
- `t12`: avg WER `4.36%`, RTF `0.074`, short clips `0.516s`
- `t6`: avg WER `4.36%`, RTF `0.070`, short clips `0.530s`
- Conclusion: only worth it if the goal is minimum latency above all else. It is extremely fast, but the error rate is high enough that it should not replace the default.

## Error pattern notes

### `whisper-small.en`

- Mostly minor errors:
  - `Concord` -> `Concorde`
  - `Raoul` -> `Raul`
  - one stray trailing quote
- These are annoying but much less severe than the gains in latency.

### `whisper-base.en`

- Real word substitutions show up:
  - `were poured in` -> `report in`
  - `why Buckingham` -> `my Buckingham`
- This matches the higher WER and makes it risky as the primary dictation model.

### `distil-large-v3.5-ct2`

- It did not fail catastrophically, but it introduced name/spelling errors such as `Raoul` -> `Ral` and still stayed in the slow-model latency class.

## Hardware utilization notes

- "Use all cores" was not universally best.
- For `distil-medium-en`, `whisper-small.en`, and `distil-large-v3.5-ct2`, 6 threads beat 12 threads.
- For `whisper-large-v3-turbo`, 12 threads beat 6 threads.
- For `whisper-base.en`, 6 and 12 threads were basically tied.
- Practical takeaway: thread count should be chosen per model, not globally forced to all logical cores.

## Recommendation

1. Keep `distil-medium-en` as the default balanced model.
2. Add `whisper-small.en` as an explicit fast option.
3. Keep `whisper-large-v3-turbo` available as the accuracy-first option.
4. Do not adopt `distil-large-v3.5-ct2` on this hardware.
5. Do not adopt `whisper-base.en` as the default, but keep it around if sub-second latency matters more than transcription quality.

## If GPU use becomes a priority

- The current repo/runtime is not using the AMD GPU.
- To test GPU seriously on this machine, the project would need a different backend or stack, for example:
  - a ROCm-capable inference path
  - a Vulkan/OpenCL path such as `whisper.cpp`
  - or a different engine entirely
- That is a separate implementation task; these results should be treated as CPU-only numbers.
