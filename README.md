# Structured Output Experiments

This repository contains experiments and demos for generating, validating, and analyzing structured outputs (primarily JSON) from language models using various prompting and schema enforcement techniques. The focus is on evaluating how well different models and libraries can produce outputs that conform to specified schemas, especially for tasks requiring structured data.

## Purpose

The main goal of this repository is to systematically benchmark and analyze the structured output capabilities of language models. It does this through a collection of scripts that:

- Generate structured data (such as JSON) from prompts.
- Validate generated data against Pydantic schemas.
- Compare and analyze the performance of different techniques and models (including OpenAI-compatible APIs, Instructor, Outlines, etc.).
- Explore error cases, edge cases, and robustness of structured output generation.

## Scripts Overview

### `pydantic_demo.py`
- **Purpose:** Tests how well language models can generate outputs that match various Pydantic schemas by prompting them with a range of scenarios (including edge cases).
- **What it does:** 
  - Defines multiple schemas (e.g., `NameYear`, `Car`, `Person`, `Book`, etc.).
  - Runs prompts for each schema and attempts to parse model outputs into the schema.
  - Tracks and prints success/failure rates and saves detailed results.
- **Use case:** Baseline for schema-conformant output using direct prompts and Pydantic validation.

### `instructor_demo.py`
- **Purpose:** Evaluates the `instructor` library (with OpenAI-compatible APIs, e.g., Groq) for structured output generation.
- **What it does:** 
  - Similar to `pydantic_demo.py`, but uses the `instructor` library for schema enforcement.
  - Connects to models via an API key.
  - Runs prompts, enforces schema, tracks retries and duration, and summarizes detailed performance statistics.
- **Use case:** Benchmarks the instructor library versus plain prompting.

### `outlines_prompting_demo.py`
- **Purpose:** Experiments with the [Outlines](https://github.com/outlines-dev/outlines) library and Hugging Face models for structured output.
- **What it does:** 
  - Loads a local or Hugging Face Transformers model.
  - Uses Outlines to enforce output structure.
  - Defines and tests the same set of schemas and prompts as the other scripts.
  - Logs and saves detailed results, including model stats and timing.
- **Use case:** Test Outlines' regex and schema-based output control.

### `structures_outlines.py`
- **Purpose:** Demonstrates code validation and structured output using the Outlines library.
- **What it does:** 
  - Shows how to use regex constraints on outputs (e.g., for sentiment classification).
  - Loads models and wraps them with Outlines for structured generation.
- **Use case:** Targeted experiments for validating code or enforcing choice constraints in outputs.

### `utils.py`
- **Purpose:** Utility functions for prompting, result formatting, and visualization.
- **What it does:** 
  - Contains helper functions like `template` (for consistent prompt formatting).
  - Provides plotting utilities for analyzing token distributions and heatmaps.
- **Use case:** Internal support for the main scripts.

## Typical Workflow

1. Choose a script (`pydantic_demo.py`, `instructor_demo.py`, or `outlines_prompting_demo.py`) depending on the model/library to benchmark.
2. Run the script to generate outputs from a set of prompts, each mapped to an expected schema.
3. The script tests each output, validates it against the schema, and records statistics (success rate, failure rate, retries, duration).
4. Results are printed and also saved to a JSON file for further analysis.

## Requirements

- Python 3.8+
- `pydantic`, `openai`, `instructor`, `outlines`, `transformers`, `dotenv`, `matplotlib` (for plotting), etc.

Add 'KEY' for groq_api_key to .env file.

## Customization

- To test new models or schemas, modify the corresponding script and/or add new schemas to the schemas section.
- Prompts and schemas are easily extensible.

## Results

Each script outputs a summary of success/failure rates, model-wise breakdowns, and saves detailed logs/results for further inspection. This allows for systematic comparison between different approaches and model capabilities for structured output.

---

**For more details, see the docstrings and comments in each script.**
