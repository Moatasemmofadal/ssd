<h1 align="center">Speculative Speculative Decoding</h1>

<h3 align="center">
  <a href="https://arxiv.org/pdf/2603.03251">Paper</a>
</h3>

<p align="center">
  <img width="1014"
       src="https://github.com/user-attachments/assets/4a38ae2d-e809-41ed-881e-fa94af820a17" />
</p>

SSD is a new LLM inference algorithm aimed at making decoding faster. This is a lightweight inference engine custom built to implement SSD. It also supports autoregressive decoding and vanilla speculative decoding as baselines, using the Llama-3 and Qwen-3 model families. 

## Setup

Requirements: Python 3.11+, CUDA >= 12.8. This code was written and tested on H100s. 

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# if `uv` is not found in this shell:
export PATH="$HOME/.local/bin:$PATH"
```

Then: 

```bash
git clone https://github.com/tanishqkumar/ssd && cd ssd
uv sync                    # core SSD deps
# uv sync --extra scripts  # add deps used by scripts/
source .venv/bin/activate
python -c "from ssd import LLM; print('ok')"
```

Set paths via environment variables. `SSD_HF_CACHE` should point to the HuggingFace **hub** directory — this is the directory that contains `models--org--name/` subdirectories (e.g. `/data/huggingface/hub`, not `/data/huggingface/`). `SSD_DATASET_DIR` should point to the directory containing the dataset subdirectories (`humaneval/`, `alpaca/`, etc).

```bash
export SSD_HF_CACHE=/path/to/huggingface/hub
export SSD_DATASET_DIR=/path/to/processed_datasets
export SSD_CUDA_ARCH=9.0   # 9.0=H100, 8.0=A100, 8.9=L40/4090
```

### Download models + datasets

If you already have the models downloaded via `huggingface-cli` or similar, you can skip straight to datasets — just make sure `SSD_HF_CACHE` points to the right place. The download scripts require the `scripts` extra: `uv sync --extra scripts`.

```bash
# models (uses SSD_HF_CACHE)
python scripts/download_from_hf.py llama

# datasets (writes to $HF_DATASETS_CACHE/processed_datasets)
export HF_DATASETS_CACHE=/path/to  # parent of SSD_DATASET_DIR
python scripts/get_data_from_hf.py --num-samples 10000
```

## Usage

Run benchmark commands from inside the `bench/` directory. Use `--all` for full eval across the four datasets. Use `python -O` for benchmarking to disable debug overhead. `--numseqs` is per-dataset, so `--numseqs 128 --all` runs 128 × 4 = 512 prompts total.

Large target (Llama-3 70B, Qwen-3 32B) runs take a few minutes for load/warmup/compile before token generation starts.

```bash
cd bench

# AR — Llama 70B, 4 GPUs
python -O bench.py --llama --size 70 --gpus 4 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# Sync spec decode — 70B target + 1B draft, 4 GPUs, k=6
python -O bench.py --llama --size 70 --gpus 4 --spec --k 6 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# Async spec decode — 70B target (4 GPUs) + 1B draft (1 GPU), k=7, f=3
python -O bench.py --llama --size 70 --gpus 5 --spec --async --k 7 --f 3 --b 1 --temp 0 --numseqs 128 --output_len 512 --all
```

Use `--qwen --size 32` for Qwen models. See `bench/bench.py` for full args. For SGLang/vLLM baselines, see `bench/README.md`, you'll have to make separate environments. 
