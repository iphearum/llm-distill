# Khmer tokenizer expansion and Qwen3 training

This project experiments with adding frequent Khmer morphemes as individual tokens to a Qwen3 tokenizer, then training the added embedding and output rows while keeping the original model weights frozen. The pipeline is split into scripts `01` through `07` so data preparation, vocabulary discovery, training, evaluation, serving, and optional Unsloth inference can be run separately.

## Hardware profiles

The default base model is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). It is the practical choice for this project on a CPU-only Windows computer. CPU training is supported, but it can be slow and needs several GB of system RAM. Start with a short run to check the pipeline before increasing training steps.

Qwen3-8B is an alternative for a machine with a compatible NVIDIA CUDA GPU, enough GPU memory, and system RAM. An Intel Iris Xe graphics device is not a CUDA GPU, so use the 0.6B CPU profile on that machine. The scripts do not train on Intel graphics.

## Before you start: create the environment

Open PowerShell in the project folder. Install a PyTorch build that matches your device first; `requirements.txt` intentionally does not select a PyTorch backend.

For the local CPU profile, use the CPU option in the [official PyTorch install selector](https://pytorch.org/get-started/locally/), then install the project dependencies:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
uv pip install fastapi uvicorn pydantic
```

If PowerShell blocks virtual environment activation, either allow local scripts for the current user or run commands with the environment's Python directly, for example `\.venv\Scripts\python.exe 01_prepare.py`.

Check the selected backend:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For an NVIDIA computer, install the CUDA build recommended by the PyTorch selector instead of the CPU wheel. Use a GPU that supports BF16 for this training script. Do not install the CPU wheel on that machine.

## 01. Prepare the corpus

Run:

```powershell
python 01_prepare.py --max-per-dataset 12000
```

This streams the configured Khmer text datasets, normalizes and cleans the text, removes duplicate records, then writes:

- `data/train.jsonl`
- `data/valid.jsonl`

`--max-per-dataset` limits the number of examples read from each source dataset. Use `--max-per-dataset 0` to process all available rows. The script also accepts `--validation-ratio` (default `0.05`) and `--seed` (default `42`).

## 02. Discover candidate Khmer tokens

Run the default 0.6B profile:

```powershell
python 02_discover.py --base Qwen/Qwen3-0.6B --top-k 512 --min-frequency 8 --max-rows 20000 --workers 6
```

The script reads `data/train.jsonl`, segments Khmer text with `khmer-segmenter`'s morpheme tokenizer, counts candidate segments, and scores them by how many base-tokenizer pieces they can replace. It shows progress bars for segmentation and candidate scoring. The output is:

- `artifacts/candidates.json`

Useful options include `--top-k`, `--min-frequency`, `--max-chars`, `--max-rows`, `--workers`, `--chunk-size`, and `--tokenizer-batch-size`. `--max-rows 0` processes the full training file. If discovery is slow, lower `--max-rows`; if the worker processes overwhelm the computer, lower `--workers`.

## 03. Build the expanded tokenizer

Run with the same base model used for discovery:

```powershell
python 03_tokenizer.py --base Qwen/Qwen3-0.6B
```

This adds the mined candidates as literal `AddedToken` entries and saves:

- `artifacts/tokenizer/`
- `artifacts/vocab_meta.json`

The script prints before/after token counts for examples including `ទោះជាយ៉ាងណាក៏ដោយ`. For example, if discovery found the six segments, the intended tokenization is `ទោះ`, `ជា`, `យ៉ាង`, `ណា`, `ក៏`, `ដោយ`. The exact result depends on the candidates selected from the corpus.

If you change the base model or discovery settings, rerun steps 02 and 03. Keep the same base model for discovery, tokenizer creation, benchmarking, and training.

## 04. Train the added token rows

First do a short CPU smoke run:

```powershell
python 04_train.py --device cpu --steps 10 --seq-len 64 --cpu-threads 4
```

This loads the base model named in `artifacts/vocab_meta.json`, initializes new token vectors from the original tokenizer's decompositions, and trains only the new input-embedding and output-projection rows. It saves the expanded model to:

- `artifacts/qwen3-khmer-expanded/`

Once the short run completes, increase `--steps` and `--seq-len` gradually. The defaults are `--steps 200` and `--seq-len 192`; longer CPU runs can take a long time. The `--output` option changes the checkpoint directory. `--lr` and `--seed` adjust learning rate and reproducibility.

For the 8B CUDA profile, use `Qwen/Qwen3-8B` consistently in steps 02 and 03, then run:

```powershell
python 04_train.py --device cuda --steps 200 --seq-len 192
```

The script reads the base model from the metadata, so changing only the training command does not convert 0.6B artifacts into an 8B run. Rebuild the candidates and tokenizer for the chosen base first. CUDA training requires a supported NVIDIA GPU; the CPU path intentionally rejects the 8B base.

## 05. Benchmark token counts and generation

After training, compare token counts on the held-out validation data:

```powershell
python 05_benchmark.py --model artifacts/qwen3-khmer-expanded --base Qwen/Qwen3-0.6B --samples 100
```

To also generate sample outputs and measure generation latency:

```powershell
python 05_benchmark.py --model artifacts/qwen3-khmer-expanded --base Qwen/Qwen3-0.6B --samples 2 --generate
```

This step needs `data/valid.jsonl`. Pass the same `--base` used in steps 02 and 03. `--samples` controls how many validation examples are measured. Token-count reduction alone does not prove the model is faster or produces better Khmer; compare matched prompts and evaluate response quality as well.

## 06. Serve the trained model

Install the API dependencies from the setup section, then copy the numbered script to a Python module name PowerShell can import:

```powershell
Copy-Item 06_serve.py serve.py
uvicorn serve:app --host 127.0.0.1 --port 8000
```

The server loads `artifacts/qwen3-khmer-expanded` by default. Check that it is running:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

To select another model directory, set `MODEL` before starting Uvicorn. `DEVICE` accepts `auto`, `cpu`, or `cuda`; `auto` selects CUDA when available and CPU otherwise.

```powershell
$env:MODEL = "artifacts/qwen3-khmer-expanded"
$env:DEVICE = "cpu"
uvicorn serve:app --host 127.0.0.1 --port 8000
```

The API implements `/v1/models` and `/v1/chat/completions`. It binds to `127.0.0.1`, so it is available only from the local computer. Add authentication and appropriate network protections before exposing a service beyond localhost.

## 07. Optional: check Unsloth inference

`07_unsloth_inference.py` is an optional CUDA-only compatibility check for loading the saved model through Unsloth in 4-bit mode. It is not needed for training or for the regular Transformers server. Install Unsloth in a separate environment with PyTorch/CUDA versions supported by its installation instructions, then run:

```powershell
python 07_unsloth_inference.py
```

This script sends tensors to CUDA, so it will not run on the Intel Iris Xe CPU setup. Unsloth compatibility with a tokenizer whose vocabulary has been expanded should be treated as experimental.

## Outputs by step

| Script | Main output |
| --- | --- |
| `01_prepare.py` | `data/train.jsonl`, `data/valid.jsonl` |
| `02_discover.py` | `artifacts/candidates.json` |
| `03_tokenizer.py` | `artifacts/tokenizer/`, `artifacts/vocab_meta.json` |
| `04_train.py` | `artifacts/qwen3-khmer-expanded/` |
| `05_benchmark.py` | Console report; optional generated samples |
| `06_serve.py` | Local HTTP API when launched with Uvicorn |
| `07_unsloth_inference.py` | Optional CUDA inference check in console |

## Current limitations

- Candidate segmentation follows the segmenter's dictionary and may miss useful boundaries for ambiguous or unknown Khmer text.
- Added tokens match literal strings. They are not BPE merge rules, and tokenization can depend on the exact text and normalization.
- Training updates only the new token embedding and output rows. Transformer layers remain frozen, so this is a research prototype rather than full Khmer adaptation.
- Fewer tokenizer pieces do not by themselves guarantee better speed, response quality, or language understanding.
- Keep the base model, tokenizer, and vocabulary metadata together. Rebuild and train the tokenizer artifacts when changing the vocabulary.
