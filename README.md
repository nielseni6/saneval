# SANEval
Suite of benchmarks for evaluating generated image composition.

## Installation

```bash
pip install -e .
```

## Authentication Setup

SANEval supports multiple VLM/LLM providers. Choose and configure the provider(s) you need:

### Google Gemini (Default)

**Option 1: Application Default Credentials (Recommended)**
```bash
gcloud auth application-default login
```

**Option 2: API Key**
```bash
export GOOGLE_API_KEY="your-api-key"
```

### OpenAI

Set your API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

### AWS Bedrock (Claude)

**Option 1: AWS CLI Configuration (Recommended)**
```bash
aws configure
```

**Option 2: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"  # or your preferred region
```

**Option 3: IAM Role**
- When running on AWS infrastructure (EC2, ECS, Lambda), IAM roles are automatically detected

## Usage

Run benchmarks on a directory of images:

```bash
python ssa/benchmark.py \
  --images-dir images/samples/attribute_binding \
  --output-dir results/attribute_binding \
  --scoring od-attr-binding
```

### Available Scoring Methods

- `od-attr-binding`: Object detection attribute binding evaluation
- `spatial`: Spatial relationship evaluation
- (Add other scoring methods as needed)

## Project Structure

```
ssa/
├── benchmark.py          # Main benchmark runner
├── providers/            # VLM/LLM provider implementations
│   ├── gemini.py        # Google Gemini
│   ├── openai.py        # OpenAI GPT
│   └── bedrock.py       # AWS Bedrock (Claude)
├── scorers/             # Scoring method implementations
├── utils/               # Utility functions
└── data/                # Data files (e.g., object categories)
```
