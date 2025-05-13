# Semantic Parser for LLM Responses

This project implements a research paper on semantic parsing for Large Language Model (LLM) responses. The system transforms natural language responses from LLMs into structured, machine-readable commands that can be used for UI automation, workflow execution, and task management.

## 🌟 Features

- **Advanced Semantic Parsing**: Utilizes dependency parsing, NER, and intent detection to understand LLM responses
- **Context-Aware Memory**: Maintains conversation history to improve parsing accuracy over time
- **Comprehensive Evaluation Framework**: Measures performance using BLEU, ROUGE, and accuracy metrics
- **Demo Application**: Includes a fully functional CLI demo for interactive testing
- **Test Suite**: Contains unit and integration tests to ensure reliability

## 🔍 Key Components

1. **Semantic Parser Core**: Transforms natural language into structured commands
   - Intent detection using zero-shot classification
   - Named entity recognition
   - Dependency parsing
   - Memory component for context retention

2. **Evaluation Framework**: Measures parser performance
   - BLEU score calculation
   - ROUGE metrics
   - Intent and entity accuracy metrics
   - Visualization tools

3. **Demo Application**: Interactive CLI for testing the system
   - Simulated LLM responses (or connect to real API)
   - Command execution visualization
   - Rich, user-friendly interface

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)
- SpaCy English language model

### Installation

1. Clone this repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_lg
   ```
3. Run the demo application:
   ```bash
   python demo_application.py
   ```

## 📊 Example Usage

### Basic Example

```python
from semantic_parser import SemanticParser

parser = SemanticParser()

# Parse a simple command
result = parser.parse("Open the email application")
print(result)
# Output: {'type': 'open', 'entity': 'email application', 'raw_text': 'Open the email application'}

# Parse a conditional command
result = parser.parse("Check the weather for tomorrow, and if it's sunny, schedule a picnic")
print(result)
# Output: {'type': 'check_condition', 'condition': 'weather for tomorrow', 'then': 'schedule a picnic', 'raw_text': 'Check the weather for tomorrow, and if it\'s sunny, schedule a picnic'}
```

### Using the Evaluation Framework

```python
from semantic_parser import SemanticParser
from evaluation_framework import SemanticParserEvaluator

parser = SemanticParser()
evaluator = SemanticParserEvaluator(parser)

test_data = [
    ("Open the email application", {"type": "open", "entity": "email application"}),
    ("Schedule a meeting for tomorrow at 10 AM", {"type": "schedule", "event_type": "meeting", "time": "tomorrow at 10 AM"})
]

results = evaluator.evaluate(test_data)
print(results["average"])
```

## 📋 Project Structure

```
semantic-parser-for-llm/
├── semantic_parser.py       # Core parser implementation
├── evaluation_framework.py  # Evaluation metrics and visualizations
├── demo_application.py      # Interactive demo application
├── test_suite.py            # Comprehensive test suite
├── requirements.txt         # Package dependencies
└── README.md                # Project documentation
```

## 🔧 Customization

The semantic parser can be customized for specific domains by:

1. Adding domain-specific intent labels
2. Extending command templates
3. Implementing custom command executors
4. Fine-tuning the intent classifier on domain data
