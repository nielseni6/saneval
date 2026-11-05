# SANEval: Open-Vocabulary Compositional Benchmarks with Failure-mode Diagnosis

[![Paper](https://img.shields.io/badge/NeurIPS%202026-Evaluations%20%26%20Datasets%20Track-blue)](https://openreview.net/pdf?id=Er9rKIjTkD)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-saneval%2Fsaneval--release-yellow)](https://huggingface.co/datasets/saneval/saneval-release)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**SANEval** (Spatial, Attribute, and Numeracy Evaluation) is an open-vocabulary compositional benchmark for text-to-image (T2I) models. It pairs a public prompt–image dataset with a modular evaluation pipeline that combines an LLM-based prompt parser, an open-vocabulary detector (YOLO-E) with LLM-driven synonym expansion and mapping, and three scorers for **attribute binding** (color, shape, texture), **spatial relations**, and **numeracy**. Every scorer emits a numerical score *and* structured diagnostic feedback (what is missing, extra, or mis-bound).

This repository contains the evaluation pipeline (the `ssa` package — *Standardized Scoring Architecture*) used to produce all results in the paper.

- **Paper:** [SANEval: Open-Vocabulary Compositional Benchmarks with Failure-mode Diagnosis](https://openreview.net/pdf?id=Er9rKIjTkD) — NeurIPS 2026 Evaluations and Datasets Track
- **Dataset:** [`saneval/saneval-release`](https://huggingface.co/datasets/saneval/saneval-release) on Hugging Face (prompts + images from six T2I models)
- **Authors:** Rishav Pramanik\*, Ian E. Nielsen\*, Jeff Smith, Saurav Pandit, Ravi P. Ramachandran, Zhaozheng Yin (\* equal contribution)

## Citation

If you use SANEval, please cite:

```bibtex
@inproceedings{pramanik2026saneval,
  title     = {{SANEval}: Open-Vocabulary Compositional Benchmarks with Failure-mode Diagnosis},
  author    = {Pramanik, Rishav and Nielsen, Ian E. and Smith, Jeff and Pandit, Saurav and Ramachandran, Ravi P. and Yin, Zhaozheng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026},
  url       = {https://openreview.net/pdf?id=Er9rKIjTkD}
}
```

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff).

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (latest version recommended)

### Setup

1. **Create and activate a virtual environment:**

```bash
# Create virtual environment
python -m venv saneval_env

# Activate the environment
source saneval_env/bin/activate  # Unix/Mac
# OR
saneval_env\Scripts\activate  # Windows
```

2. **Install the package in editable mode:**

```bash
pip install -e .
```

This will automatically install all required dependencies from `requirements.txt`.

## Dataset

The SANEval prompt sets and the images used for every result in the paper are released at
[`saneval/saneval-release`](https://huggingface.co/datasets/saneval/saneval-release).
Each record has five fields: `prompt`, `model` (one of six T2I models), `split` (`simple` = template-generated, author-reviewed prompts; `hard` = fully human-written prompts), `type` (`spatial`, `numeracy`, `color`, `shape`, `texture`), and `image`.

The benchmark runner reads prompts **from image filenames**, using the layout `<images-dir>/<prompt text>_<6-digit index>.png` (see `images/samples/` for examples). The snippet below streams one model/split from the Hugging Face release into that layout:

```python
import os
from datasets import load_dataset

MODEL, SPLIT = "Imagen 4.0 Ultra", "simple"
ds = load_dataset("saneval/saneval-release", split="train", streaming=True)
ds = ds.filter(lambda r: r["model"] == MODEL and r["split"] == SPLIT)

counters = {}
for r in ds:
    out_dir = os.path.join("images", "imagen4ultra_simple", r["type"])
    os.makedirs(out_dir, exist_ok=True)
    i = counters[r["type"]] = counters.get(r["type"], 0) + 1
    r["image"].save(os.path.join(out_dir, f"{r['prompt']}_{i:06d}.png"))
```

Then point `ssa/benchmark.py` at each category directory (`spatial` → `--scoring spatial`, `numeracy` → `--scoring numeracy`, `color`/`shape`/`texture` → `--scoring od_attr_binding`), as shown under [Usage](#usage).

To evaluate your own T2I model, generate an image for each prompt in the dataset and save it with the same filename convention.

## Authentication Setup

SANEval supports multiple VLM/LLM providers. The paper's default backbone for both the LLM and VLM components is Gemini 2.5 Flash; Llama 4 Maverick (via AWS Bedrock) is used for the backbone-swap experiments.

### Google Gemini

**Option 1: Application Default Credentials (Recommended)**
```bash
gcloud auth application-default login
```

**Option 2: API Key**
```bash
export GOOGLE_API_KEY="your-api-key"
```

### AWS Bedrock (Llama 4 Models)

SANEval supports Llama 4 Maverick via AWS Bedrock. See [docs/aws_setup.md](docs/aws_setup.md) for detailed setup instructions.

**Quick Start:**

```bash
# Option 1: Environment Variables (CI/CD)
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Option 2: AWS Profile (Local Development)
export AWS_PROFILE=your_profile_name
export AWS_DEFAULT_REGION=us-east-1
```

**Supported Models:**
- `bedrock/llama-4-maverick-17b-instruct` - Llama 4 Maverick (17B parameters)
  - Vision support: ✅ Yes (multimodal)
  - Structured output: ✅ Via prompt engineering
  - Max tokens: 2048 (configurable)

**Usage Example:**

```python
from ssa.vlm import Vlm
from ssa.providers.bedrock import LLAMA_4_MAVERICK
from PIL import Image

vlm = Vlm(LLAMA_4_MAVERICK)
image = Image.open("photo.jpg")
response = vlm.call("Describe this image", image=image)
```

For more examples, see `examples/bedrock_llama_examples.py`.

### Selecting backbones

The LLM, VLM, and detector backbones are configured through environment variables (defaults shown):

```bash
export SSA_DEFAULT_VLM="gemini/2.5-flash"   # VLM-as-judge for attribute reading
export SSA_DEFAULT_LLM="gemini/2.5-flash"   # prompt parsing, synonym expansion/mapping, similarity scoring
export SSA_DEFAULT_OD="yolo/e"              # open-vocabulary detector (also: yolo/v11, yolo/v12, yolo/world)
```

To reproduce the Llama-4 backbone-swap results, set `SSA_DEFAULT_VLM` and `SSA_DEFAULT_LLM` to `bedrock/llama-4-maverick-17b-instruct`.

## Usage

### Command Line

Run benchmarks on a directory of images using different scoring methods. Each run writes `run_results.json` (per-prompt scores and diagnostic feedback), `aggregates.json`, and `reasoning_trace.json` to `--output-dir`.

#### Numeracy Evaluation
Evaluate counting and object quantities in images:

```bash
python ssa/benchmark.py \
  --images-dir images/samples/numeracy \
  --output-dir results/numeracy \
  --scoring numeracy
```

#### Spatial Relationship Evaluation
Evaluate spatial relationships between objects:

```bash
python ssa/benchmark.py \
  --images-dir images/samples/spatial \
  --output-dir results/spatial \
  --scoring spatial
```

#### Attribute Binding Evaluation
Evaluate object attributes and their bindings:

```bash
python ssa/benchmark.py \
  --images-dir images/samples/attribute_binding \
  --output-dir results/attribute_binding \
  --scoring od_attr_binding
```

Run `python ssa/benchmark.py --help` for the full list of options, including `--bench-config` and `--execution-config` overrides.

### VS Code Debugging

A debug configuration is included for VS Code:

1. Open the project in VS Code
2. Ensure `saneval_env` is created and activated
3. Go to Run and Debug (Ctrl+Shift+D / Cmd+Shift+D)
4. Select "Benchmark: Attribute Binding" from the dropdown
5. Press F5 to start debugging

The debug configuration automatically uses the `saneval_env` Python interpreter.

### Available Scoring Methods

- `numeracy`: Count and quantity evaluation (e.g., "two cats", "three red apples")
- `spatial`: Spatial relationship evaluation (e.g., "a cat next to a dog", "a bird on the top of a tree")
- `od_attr_binding`: Object detection attribute binding evaluation for color, shape, and texture (e.g., "a red car and a blue house")

## Project Structure

```
ssa/
├── benchmark.py          # Main benchmark CLI entry point
├── benchmark_runner.py   # Loads images, extracts prompts, runs scorers, aggregates results
├── providers/            # VLM/LLM provider implementations
│   ├── gemini.py         # Google Gemini
│   └── bedrock.py        # AWS Bedrock (Llama 4 Maverick)
├── scorers/              # Scoring method implementations (numeracy, spatial, attribute binding)
├── benchmark_scorers/    # Benchmark-level scorer wrappers
├── scoring/              # Scoring runner and adapters
├── prompts.py            # Prompt parsing / prompt-set handling
├── od.py                 # Open-vocabulary object detection (YOLO-E) with synonym expansion and mapping
├── vlm.py                # VLM-as-judge interface
├── utils/                # Utility functions (metrics, images, costs, logging, ...)
└── data/pred_classes/    # Object category lists
examples/                 # Usage examples
docs/aws_setup.md         # AWS Bedrock setup guide
tests/                    # Unit and integration tests
```

## Supported Models

### Google Gemini Models
- `gemini/2.5-flash` - Fast and efficient (default)
- `gemini/2.5-pro` - Most capable
- `gemini/2.5-flash-lite-preview` - Lightweight preview

### AWS Bedrock Models
- `bedrock/llama-4-maverick-17b-instruct` - Llama 4 Maverick (17B parameters)
  - Vision support: ✅
  - Structured output: ✅ (prompt engineering)
  - Cost tracking: ✅

For detailed AWS setup instructions, see [docs/aws_setup.md](docs/aws_setup.md).

## License

The code in this repository is released under the [Apache License 2.0](LICENSE). The SANEval dataset is released separately under the MIT license on Hugging Face.

## Contact

- Ian E. Nielsen — iannielsen [at] live.com
- Rishav Pramanik — rpramanik [at] cs.stonybrook.edu

Please open a GitHub issue for bugs or feature requests.
