# VibeVoice4macOS — VibeVoice on Apple Silicon (macOS)

This fork adds a **one-file, self-contained setup & runner** for the full VibeVoice-Large [VibeVoice](https://github.com/WhoPaidItAll/VibeVoice) so you can launch the Gradio demo or do simple CLI inference **locally on macOS Apple Silicon**—no CUDA, no sudo, no global installs.

> ✅ Apple Silicon (arm64) + Python ≥ 3.10

> ✅ Everything lives inside this repo (no external runtime folder)

> ✅ Uses PyTorch **MPS** when available (otherwise CPU)

> ✅ Resumes & verifies large sharded model downloads

> ✅ Optional Hugging Face token auto-loaded from `./.env` or `./.hf_token`

---
<img width="1517" height="932" alt="image" src="https://github.com/user-attachments/assets/3e4aa10e-8b36-4eb4-b72c-f761ab0fbfd7" />

---

## What’s included in this fork

* `vibevoice_mac_arm64.sh` — the all-in-one installer/runner for macOS:

  * Creates a local venv
  * Clones the upstream repo
  * Downloads the chosen model with resume & shard verification
  * Provides a portable `ffmpeg` if needed (or `--allow-brew`)
  * Runs the **official** Gradio demo via a bootstrap that:

    * forces **SDPA** attention (avoids FlashAttention)
    * **never** dispatches to CUDA
    * prefers **MPS** on Apple GPUs

> Upstream model code, demos, and assets remain under their original directories; this script only orchestrates a Mac-friendly setup.

---

## Requirements

* macOS on **Apple Silicon** (`arm64`)
* **Python 3.10+** available as `python3`
* Internet for first run (pip, git, model download)
* Several GB of free disk (models can be large)

---

## Quick start

```bash
# 0) Fork this repo

# 1) Clone your fork (this repo)
git clone https://github.com/<your-username>/vibevoice_mac.git
cd vibevoice_mac

# 2) Make the setup script executable
chmod +x setup.sh

# 3) (Recommended) Add your Hugging Face token once
printf 'HF_TOKEN=hf_your_token_here\n' > ./.env

# 4) Run the Gradio demo (default model; local UI on port 7860)
bash vibevoice_mac_arm64.sh --demo

# 5) Share the demo publicly (Gradio share URL)
bash vibevoice_mac_arm64.sh --demo --share

# 6) Try a smaller model if the Large one is heavy
bash vibevoice_mac_arm64.sh --model microsoft/VibeVoice-1.5B --demo
```

---

## Hugging Face token (for gated/private models)

The script automatically picks a token from:

1. Existing env (`HF_TOKEN`)
2. `./.env` (e.g. `HF_TOKEN=hf_xxx`)
3. `./.hf_token` (file containing only the token string)

Examples:

```bash
# One-time setup (preferred)
cat > ./.env <<'EOF'
HF_TOKEN=hf_your_token_here
EOF

# Or pass inline per run
HF_TOKEN=hf_your_token_here bash vibevoice_mac_arm64.sh --demo
```

> **Never commit your token.** It lives in `./.env` (gitignored by default).

---

## Basic usage

### Launch Gradio demo

```bash
# Default model: aoi-ot/VibeVoice-Large
bash vibevoice_mac_arm64.sh --demo

# Public sharing
bash vibevoice_mac_arm64.sh --demo --share

# Different model
bash vibevoice_mac_arm64.sh --model microsoft/VibeVoice-1.5B --demo
```

### CLI-style inference

```bash
bash vibevoice_mac_arm64.sh --infer
# Output: ./outputs/sample_out.wav
```

### Clean everything

```bash
bash vibevoice_mac_arm64.sh --clean --force
```

### Useful env/flags

```bash
# Change the demo port
PORT=7861 bash vibevoice_mac_arm64.sh --demo

# Allow Homebrew ffmpeg (if missing)
bash vibevoice_mac_arm64.sh --allow-brew --demo
```

---

## Models (quick guide)

| Model               | Context | Generation | Link                                                                                                 |
| ------------------- | ------: | ---------: | ---------------------------------------------------------------------------------------------------- |
| **VibeVoice-1.5B**  |     64K |   \~90 min | [https://huggingface.co/microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)   |
| **VibeVoice-Large** |     32K |   \~45 min | [https://huggingface.co/microsoft/VibeVoice-Large](https://huggingface.co/microsoft/VibeVoice-Large) |

* Default in this script: `aoi-ot/VibeVoice-Large` (change via `--model ...`)
* Some models are **gated/private** → you’ll need a valid **HF token**

---

## What the script does (under the hood)

* Uses the project directory and creates:

  * `.venv/` (local Python virtualenv)
  * `_cache/` (HF/Torch/Transformers caches)
  * `models/` (downloaded model files)
  * `tools/ffmpeg/ffmpeg` (portable binary if needed)
  * `VibeVoice/` (upstream repo)
  * `outputs/` (audio from CLI path)
* Pins all HF caches inside the project folder (no global cache usage).
* Verifies shard completeness from `model.safetensors.index.json` and **resumes** if any pieces are missing.
* Bootstraps the demo to **force SDPA** and **avoid CUDA** on macOS.

---

## Troubleshooting

* **401 / “Repository Not Found” / gated model**
  Add a valid **HF token** (see token section above) and make sure the model grants your account access.

* **Missing shard error (e.g., `model-00002-of-00010.safetensors`)**
  Re-run the script; downloads resume and shards are re-verified. Also check you have enough free disk space.

* **“Torch not compiled with CUDA enabled”**
  Expected on macOS. The script never tries CUDA and forces SDPA/MPS/CPU.

* **Port already in use**
  `PORT=7861 bash vibevoice_mac_arm64.sh --demo`

* **ffmpeg not found**
  Script provides a portable `ffmpeg`; or pass `--allow-brew` to install via Homebrew.

---

## Notes & tips

* **Performance**: MPS is faster than CPU, but still slower than high-end NVIDIA GPUs. For smoother UX, try `--model microsoft/VibeVoice-1.5B`.
* **Attention backend**: The demo code prefers FlashAttention on CUDA; this fork **forces SDPA** (safe on MPS/CPU). Audio quality may differ from CUDA+FA2 runs.

---

## VibeVoice-7B parameter reference

This section lists the knobs you can use with the local VibeVoice-7B model. "Config-only" means edit the model config file and reload; changing architecture fields can make the weights incompatible.

## VibeVoice-specific generation args
_Directly accepted by .generate_

### cfg_scale
Classifier-Free Guidance scale for speech diffusion. Higher values push speech closer to the conditioning text/voice but can reduce naturalness. Set with `model.generate(..., cfg_scale=3.0, ...)`. Settable at runtime.

### return_speech
If False, the generate call skips decoding audio and returns only token sequences. Set with `model.generate(..., return_speech=False, ...)`. Settable at runtime.

### speech_tensors
Optional speech input for voice cloning. Supply waveform tensors and matching masks; used to inject speech embeddings. Set with `model.generate(..., speech_tensors=..., speech_masks=..., speech_input_mask=...)`. Settable at runtime.

### speech_masks
Boolean mask for `speech_tensors` indicating valid frames/segments. Set with `model.generate(..., speech_masks=...)`. Settable at runtime.

### speech_input_mask
Boolean mask that marks positions in the text sequence to insert the speech embeddings. Set with `model.generate(..., speech_input_mask=...)`. Settable at runtime.

### negative_prompt_ids
Accepted by the method signature but currently unused in the implementation, so it has no effect. Not settable in practice without code changes.

### negative_prompt_attention_mask
Accepted by the method signature but currently unused in the implementation, so it has no effect. Not settable in practice without code changes.

### audio_streamer
Optional `AudioStreamer`/`AsyncAudioStreamer` instance for streaming chunks during generation. Set with `model.generate(..., audio_streamer=streamer, ...)`. Settable at runtime.

### stop_check_fn
Optional callable that returns True to stop generation early (used in the loop). Set with `model.generate(..., stop_check_fn=my_fn, ...)`. Settable at runtime.

### max_length_times
Caps generated length to `max_length_times * input_length` (in tokens). Set with `model.generate(..., max_length_times=6, ...)`. Settable at runtime.

### verbose
Enables progress prints from the generation loop. Set with `model.generate(..., verbose=True, ...)`. Settable at runtime.

### refresh_negative
Controls whether the negative prompt cache is refreshed when diffusion starts. Set with `model.generate(..., refresh_negative=False, ...)`. Settable at runtime.

### parsed_scripts
Accepted but unused; currently no effect. Not settable without code changes.

### all_speakers_list
Accepted but unused; currently no effect. Not settable without code changes.

## Standard Transformers generation args
_Supported via GenerationConfig/kwargs_

### max_length
Total sequence length cap (prompt + generated). Set via `model.generate(..., max_length=...)` or `GenerationConfig(max_length=...)`. Settable at runtime.

### max_new_tokens
Maximum number of new tokens to generate. Set via `model.generate(..., max_new_tokens=...)` or `GenerationConfig(max_new_tokens=...)`. Settable at runtime.

### min_length
Minimum total length before EOS is allowed. Set via `model.generate(..., min_length=...)` or `GenerationConfig(min_length=...)`. Settable at runtime.

### do_sample
Enable sampling (True) vs greedy/beam search. Set via `model.generate(..., do_sample=True)`. Settable at runtime.

### temperature
Sampling temperature; higher increases randomness. Requires `do_sample=True`. Set via `model.generate(..., temperature=...)`. Settable at runtime.

### top_k
Top-k sampling cutoff. Requires `do_sample=True`. Set via `model.generate(..., top_k=...)`. Settable at runtime.

### top_p
Nucleus sampling cutoff. Requires `do_sample=True`. Set via `model.generate(..., top_p=...)`. Settable at runtime.

### num_beams
Beam search width. Set via `model.generate(..., num_beams=...)`. Settable at runtime.

### length_penalty
Beam search length penalty. Set via `model.generate(..., length_penalty=...)`. Settable at runtime.

### repetition_penalty
Penalize repeated tokens. Set via `model.generate(..., repetition_penalty=...)`. Settable at runtime.

### no_repeat_ngram_size
Forbid repeated n-grams of this size. Set via `model.generate(..., no_repeat_ngram_size=...)`. Settable at runtime.

### early_stopping
Stop beam search when all beams are finished. Set via `model.generate(..., early_stopping=True)`. Settable at runtime.

### eos_token_id
Token ID that ends generation. Set via `model.generate(..., eos_token_id=...)` or `GenerationConfig(eos_token_id=...)`. Settable at runtime.

### bos_token_id
Token ID used as BOS. Set via `model.generate(..., bos_token_id=...)` or `GenerationConfig(bos_token_id=...)`. Settable at runtime.

### pad_token_id
Padding token ID. Set via `model.generate(..., pad_token_id=...)` or `GenerationConfig(pad_token_id=...)`. Settable at runtime.

### logits_processor
Custom `LogitsProcessorList` to modify token scores. Set via `model.generate(..., logits_processor=...)`. Settable at runtime.

### stopping_criteria
Custom `StoppingCriteriaList` to stop generation. Set via `model.generate(..., stopping_criteria=...)`. Settable at runtime.

### prefix_allowed_tokens_fn
Callable that restricts allowed tokens by prefix. Set via `model.generate(..., prefix_allowed_tokens_fn=...)`. Settable at runtime.

## Diffusion/CFG-related settings
_Model config and inference settings_

### ddpm_num_inference_steps
Number of diffusion steps at inference time. Set at runtime with `model.set_ddpm_inference_steps(n)` or by editing `./models/VibeVoice-7B/config.json` before load.

### ddpm_batch_mul
Diffusion micro-batch multiplier. Set at runtime with `model.model.diffusion_head.config.ddpm_batch_mul = n`.

### ddpm_beta_schedule
Schedule type for diffusion (for example, `cosine`). Config-only in `./models/VibeVoice-7B/config.json`; changing it without retraining is not supported.

### ddpm_num_steps
Total training diffusion steps. Config-only; changing it without retraining is not supported.

### prediction_type
Diffusion prediction type (for example, `v_prediction`). Config-only.

### diffusion_type
Diffusion family (for example, `ddpm`). Config-only.

### head_layers
Number of diffusion head layers. Config-only.

### head_ffn_ratio
FFN ratio in diffusion head. Config-only.

### latent_size
Diffusion latent size. Config-only.

### hidden_size
Diffusion head hidden size. Config-only.

### rms_norm_eps
RMSNorm epsilon in diffusion head. Config-only.

### speech_vae_dim
Speech VAE dimension for diffusion head. Config-only.

## Tokenizer/processor input settings
_VibeVoiceProcessor.__call__ inputs_

### text
Text or script input. Can be a string, list of strings, or a path to a `.txt`/`.json` file. Set with `processor(text=..., ...)`.

### voice_samples
Optional voice references (paths or numpy arrays). Set with `processor(text=..., voice_samples=...)`.

### padding
Enable padding; can be `True`, `False`, `longest`, or `max_length`. Set with `processor(..., padding=...)`.

### truncation
Enable truncation; can be `True`/`False` or a strategy. Set with `processor(..., truncation=...)`.

### max_length
Max token length for the processor output. Set with `processor(..., max_length=...)`.

### return_tensors
Framework for tensors (for example, `pt`). Set with `processor(..., return_tensors="pt")`.

### return_attention_mask
Include attention masks. Set with `processor(..., return_attention_mask=True)`.

_Config-only processor settings (from `./models/VibeVoice-7B/preprocessor_config.json`)_

### speech_tok_compress_ratio
Tokenizer compression ratio used in speech processing. Config-only.

### db_normalize
Whether to normalize audio loudness. Config-only.

### audio_processor.sampling_rate
Target sampling rate for audio (24k default). Config-only.

### audio_processor.normalize_audio
Whether to normalize waveforms. Config-only.

### audio_processor.target_dB_FS
Target loudness for normalization. Config-only.

### audio_processor.eps
Epsilon for numerical stability in normalization. Config-only.

### processor_class
Processor class name. Config-only.

### language_model_pretrained_name
Base LLM name (Qwen2.5-7B). Config-only.

## Model loading settings
_Applies when calling from_pretrained_

### torch_dtype
Model dtype (`torch.float16` or `torch.bfloat16`). Set via `from_pretrained(..., torch_dtype=...)`.

### device_map
Device placement (`auto`, `cpu`, `mps`, or a dict). Set via `from_pretrained(..., device_map=...)`.

### attn_implementation
Attention backend (`flash_attention_2` or `sdpa`). Set via `from_pretrained(..., attn_implementation=...)`.

## Responsible use (risks & limitations)

High-quality synthetic speech can be misused (impersonation, fraud, disinformation). Use responsibly and comply with all applicable laws and policies. Disclose AI-generated content where appropriate.

Other known limitations (from upstream notes):

* English/Chinese are the strongest languages; others may be unstable.
* Cross-lingual transfer can be inconsistent.
* Spontaneous music/background sounds may appear depending on prompts.
* No explicit support for overlapping speech.

This project is intended for research & experimentation.

---

## Acknowledgements

* **Upstream**: [WhoPaidItAll/VibeVoice](https://github.com/WhoPaidItAll/VibeVoice)
* Hugging Face ecosystem (transformers, accelerate, huggingface\_hub)
* PyTorch MPS on Apple Silicon

---

## License

* The **vibevoice\_mac\_arm64.sh** helper script in this fork: MIT (or match upstream—add a `LICENSE` file accordingly).
* **VibeVoice** code/models: their respective upstream licenses & model cards apply.

---

### Repository layout

```
.
├─ vibevoice_mac_arm64.sh    # ← this script (macOS setup & runner)
├─ generate.py               # optional local generator wrapper
├─ VibeVoice/                # upstream repo code (cloned by the script at runtime)
├─ .venv/                    # local venv (created by the script)
├─ _cache/                   # HF/Torch/Transformers caches
├─ models/                   # local model files
├─ outputs/                  # generated audio
├─ tools/                    # bundled tools (ffmpeg)
└─ README.md                 # this file
```

> Runtime folders live inside this repo; keep them gitignored to avoid committing large files or secrets.
