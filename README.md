# SANEval
Suite of benchmarks for evaluating generated image composition.

## Installation

### Prerequisites
- Python 3.8 or higher
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

## Authentication Setup

SANEval uses Google Gemini as its VLM/LLM provider.

### Google Gemini

**Option 1: Application Default Credentials (Recommended)**
```bash
gcloud auth application-default login
```

**Option 2: API Key**
```bash
export GOOGLE_API_KEY="your-api-key"
```

## Usage

### Command Line

Run benchmarks on a directory of images:

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

- `od_attr_binding`: Object detection attribute binding evaluation
- `spatial`: Spatial relationship evaluation
- `numeracy`: Numeracy evaluation

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
