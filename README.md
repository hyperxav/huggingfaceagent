# Hugging Face Dummy Agent

This is a simple implementation of a conversational agent using Hugging Face's transformers library. The agent maintains a conversation history and uses a text generation pipeline to generate responses.

## Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The `DummyAgent` class provides a simple interface for text-based conversations. Here's a basic example:

```python
from transformers import pipeline
from dummy_agent import DummyAgent

# Initialize a text generation pipeline
text_pipeline = pipeline("text-generation", model="gpt2")

# Create an instance of DummyAgent
agent = DummyAgent(text_pipeline)

# Have a conversation
response = agent.run("Hello! How are you?")
print("Agent:", response)
```

You can also run the example script directly:
```bash
python example.py
```

## Features

- Simple conversation interface
- Maintains conversation history
- Uses Hugging Face's transformers library for text generation
- Customizable with different language models

## Requirements

- transformers>=4.34.0
- torch>=2.0.0
- accelerate>=0.21.0
- langchain>=0.0.27
- python-dotenv>=1.0.0

## License

This project is licensed under the MIT License - see the LICENSE file for details. 