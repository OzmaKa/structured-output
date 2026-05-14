
# Structured Output, Agents & Evaluation

Three Python projects demonstrating core AI engineering patterns:
getting reliable structured data from LLMs, building a multi-step
agent that chains tools, and evaluating + red-teaming an AI system.

---

## Projects

### reviewer.py

A code reviewer that always returns structured JSON.

* Forces JSON output via system prompt
* Validates response shape and types with Pydantic
* Computes `approved` automatically via `model_validator` when score >= 7
* Handles markdown stripping — Cohere sometimes wraps JSON in backticks

### agent.py

A persistent multi-step agent using function calling.

* Two tools: `review_code()` and `calculate()`
* LLM reads the user message and decides which tool(s) to call
* Chains multiple tool calls in a single response — no hardcoded routing
* Loops until the LLM produces a final answer or hits 10 iterations
* Safe math evaluation using `ast` module instead of `eval()`
* Stays alive between inputs — type `exit` to quit

### eval_harness.py

A complete evaluation pipeline for the code reviewer.

* **Test Suite** — 5 test cases with condition-based assertions (not exact values)
* **LLM-as-Judge** — a second LLM rates helpfulness and accuracy of each review
* **Red-Team** — 7 adversarial inputs including prompt injection, empty input,
  syntax errors, and deeply nested code

#### Vulnerabilities found and fixed via red-teaming:

| Attack                                    | Result                               | Fix                          |
| ----------------------------------------- | ------------------------------------ | ---------------------------- |
| Empty input                               | Model hallucinated score=10          | Guard clause before API call |
| Prompt injection via `# SYSTEM:`comment | Model returned score=10 for bad code | Hardened system prompt       |

---

## Stack

* Python 3.12
* Cohere API — `command-r-plus-08-2024`
* Pydantic — schema validation and computed fields
* python-dotenv — environment variables

---

## Setup

1. Install dependencies

```bash
pip install cohere pydantic python-dotenv
```

2. Create a `.env` file

```bash
COHERE_API_KEY=your_key_here
```

---

## Usage

Run the code reviewer:

```bash
python reviewer.py
```

Run the multi-step agent (persistent REPL):

```bash
python agent.py
```

Run the full evaluation harness:

```bash
python eval_harness.py
```

---

## Key Lessons

* LLM test suites assert  **conditions** , not exact values — outputs are non-deterministic
* `eval()` is a security hole — use `ast` with a whitelist for math evaluation
* Prompt injection is real — always test adversarial inputs before shipping
* The agent loop needs a `tool_call_id` for every tool result or Cohere returns 400
* Low temperature (0.1–0.2) gives more consistent structured outputs
* Cohere sometimes wraps JSON in markdown backticks even when told not to — always strip defensively
