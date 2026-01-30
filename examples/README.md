# SANEval Examples

This directory contains practical examples demonstrating how to use SANEval for various tasks.

## Overview

- **basic_evaluation.py** - Simple numeracy and spatial evaluation
- **custom_scorer.py** - Creating a custom scorer
- **benchmark_workflow.py** - Running complete benchmarks
- **advanced_config.py** - Advanced configuration options

## Running Examples

1. Ensure SANEval is installed:
```bash
pip install -e .
```

2. Set up your API keys:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

3. Run an example:
```bash
python examples/basic_evaluation.py
```

## Prerequisites

Most examples require:
- Python 3.8+
- API key for Gemini or OpenAI
- Sample images (some examples use placeholder URLs)

## Getting Help

If you encounter issues:
1. Check the main [README](../README.md) for installation instructions
2. Review [CONTRIBUTING](../CONTRIBUTING.md) for development setup
3. See [ARCHITECTURE](../ARCHITECTURE.md) for system design details
4. Open an issue on GitHub

## Example Data

Some examples expect sample images. You can:
- Use your own images
- Download sample datasets
- Generate images with text-to-image models


## Example Commands

### Using Llama

#### Spatial:

```bash
python ssa/benchmark.py \
    --images-dir images/samples/spatial \
    --output-dir results/llama_spatial \
    --scoring spatial \
    --bench-config spatial.llm=bedrock/llama-4-maverick-17b-instruct
```

#### Numeracy:

```bash
python ssa/benchmark.py \
    --images-dir images/samples/numeracy \
    --output-dir results/llama_numeracy \
    --scoring numeracy \
    --bench-config numeracy.llm=bedrock/llama-4-maverick-17b-instruct
```

#### OD Attribute Binding:

```bash
python ssa/benchmark.py \
    --images-dir images/samples/attribute_binding \
    --output-dir results/llama_od_attr_binding \
    --scoring od_attr_binding \
    --bench-config od_attr_binding.vlm_model=bedrock/llama-4-maverick-17b-instruct \
    --bench-config od_attr_binding.llm_model=bedrock/llama-4-maverick-17b-instruct
```

### Using Google Gemini (Default Models)

These examples use Google Gemini models with an API key. Set your API key first:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

#### Spatial (Gemini):

```bash
python ssa/benchmark.py \
    --images-dir images/samples/spatial \
    --output-dir results/spatial \
    --scoring spatial
```

#### Numeracy (Gemini):

```bash
python ssa/benchmark.py \
    --images-dir images/samples/numeracy \
    --output-dir results/numeracy \
    --scoring numeracy
```

#### OD Attribute Binding (Gemini):

```bash
python ssa/benchmark.py \
    --images-dir images/samples/attribute_binding \
    --output-dir results/attribute_binding \
    --scoring od_attr_binding
```