# SANEval
Suite of benchmarks for evaluating generated image composition.

## Installation

### Automatic Setup (Recommended)

The setup script will automatically create a virtual environment for you:

```bash
# Pre-setup (if pip is not updated)
pip install --upgrade pip 

# Step 1: Run setup to create the virtual environment
python setup.py install

# Step 2: Activate the virtual environment
source saneval_env/bin/activate  # Unix/Mac
# OR
saneval_env\Scripts\activate  # Windows

# Step 3: Install the package in editable mode
pip install -e .
```

### Manual Setup

If you prefer to manage your own virtual environment:

```bash
# Create and activate your own virtual environment
python -m venv myenv
source myenv/bin/activate  # Unix/Mac

# Install the package
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

### Command Line

Run benchmarks on a directory of images:

```bash
python ssa/benchmark.py \
  --images-dir images/samples/attribute_binding \
  --output-dir results/attribute_binding \
  --scoring od-attr-binding
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
