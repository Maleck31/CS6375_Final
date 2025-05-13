# Semantic Parser for LLM Responses

This project implements a semantic parser for translating LLM-generated responses into structured, machine-readable commands that can be executed automatically.

## Requirements

The implementation requires the following Python packages:

```
spacy>=3.4.0
scikit-learn>=1.0.0
transformers>=4.25.0
nltk>=3.7
rich>=12.0.0
pyautogui>=0.9.53
numpy>=1.20.0
matplotlib>=3.5.0
torch>=1.13.0
requests>=2.28.0
rouge>=1.0.1
```

Additionally, you'll need to download the SpaCy English language model:

```bash
python -m spacy download en_core_web_lg
```

## Installation Guide

### 1. Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Required Packages

```bash
# Download SpaCy language model
python -m spacy download en_core_web_lg

# Download necessary NLTK data
python -c "import nltk; nltk.download('punkt')"
```
Alternatively, you can pass these as command-line arguments when running the application.

## Running the Application

### CLI Demo

```bash
python demo_application.py
```

### Running Tests

```bash
# Run all tests
python test_suite.py

# Run specific test class
python -m unittest test_suite.TestSemanticParser

# Run specific test method
python -m unittest test_suite.TestSemanticParser.test_basic_commands
```

### Running the Evaluator

```bash
python evaluation_framework.py
```

## Project Structure

- `semantic_parser.py`: Core implementation of the semantic parser
- `evaluation_framework.py`: Framework for evaluating parser performance
- `demo_application.py`: Interactive demo application
- `test_suite.py`: Comprehensive test suite
- `requirements.txt`: Package dependencies

## Example Usage

```python
from semantic_parser import SemanticParser

# Initialize the parser
parser = SemanticParser()

# Parse an LLM response
llm_response = "Check the weather for tomorrow, and if it's sunny, schedule a picnic"
structured_command = parser.parse(llm_response)

print(structured_command)
```