# SANEval
Suite of benchmarks for evaluating generated image composition.

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

This will automatically install all required dependencies from requirements.txt.

Note: You will also need to generate the images before using this script. Prompt sets used to generate our paper results are linked [[SANEval-Simple](https://huggingface.co/datasets/nielseni6/SANEval-Simple), [SANEval-Hard](https://huggingface.co/datasets/nielseni6/SANEval-Hard)]

## Authentication Setup

SANEval supports multiple VLM/LLM providers:

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

SANEval now supports Llama 4 Maverick via AWS Bedrock. See [docs/aws_setup.md](docs/aws_setup.md) for detailed setup instructions.

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

## Usage

### Command Line

Run benchmarks on a directory of images using different scoring methods:

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
- `od_attr_binding`: Object detection attribute binding evaluation (e.g., "a red car and a blue house")

## Project Structure

```
ssa/
├── benchmark.py          # Main benchmark runner
├── providers/            # VLM/LLM provider implementations
│   └── gemini.py        # Google Gemini
├── scorers/             # Scoring method implementations
├── utils/               # Utility functions
└── data/                # Data files (e.g., object categories)
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

