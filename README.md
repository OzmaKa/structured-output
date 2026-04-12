# Structured Output & Function Calling

Two Python projects demonstrating how to get reliable, typed, 
structured data from LLMs — and how to build an agent that 
decides which tool to call based on user input.

## Projects

### reviewer.py
A code reviewer that always returns structured JSON.
- Forces JSON output via system prompt
- Validates response shape and types with Pydantic
- Computes `approved` automatically based on score

### agent.py
A multi-tool agent using function calling.
- Two tools: `review_code()` and `calculate()`
- LLM reads the user message and decides which tool to call
- No hardcoded routing — the model figures it out

## Stack

- Python 3.12
- Cohere API — chat and generation
- Pydantic — schema validation
- python-dotenv — environment variables

## Setup

1. Install dependencies
    pip install cohere pydantic python-dotenv

2. Create a .env file
    COHERE_API_KEY=your_key_here

## Usage

Run the code reviewer:
    python reviewer.py

Run the function calling agent:
    python agent.py
