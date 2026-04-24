from dotenv import load_dotenv
import cohere
import os
import json
from pydantic import BaseModel, model_validator

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

# ── Pydantic model for code review output ─────────────────────────────────────
class CodeReview(BaseModel):
    score: int
    issues: list[str]
    suggestion: str
    approved: bool = False

    @model_validator(mode='after')
    def set_approved(self):
        self.approved = self.score >= 7
        return self

# ── Tool definitions — the LLM reads these to decide which tool to call ───────
tools = [
    {
        "type": "function",
        "function": {
            "name": "review_code",
            "description": "Reviews code and returns a score, issues, and suggestion. Use this when the user asks to review, analyze, or check code quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to review"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a math expression and returns the result. Use this for any arithmetic, powers, or numerical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '847 * 293' or '3 ** 8'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ── Tool implementations ───────────────────────────────────────────────────────
def calculate(expression: str) -> str:
    # replace ^ with ** so 3^8 works as expected
    expression = expression.replace("^", "**")
    result = eval(expression)
    return str(result)

def review_code(code: str) -> dict:
    co = cohere.ClientV2(api_key)

    answer = co.chat(
        model="command-r-plus-08-2024",
        messages=[
            {
                "role": "system",
                "content": """You are a code reviewer. Return only valid JSON:
{
  "score": integer 1-10,
  "issues": [list of strings],
  "suggestion": "single string",
  "approved": boolean
}"""
            },
            {
                "role": "user",
                "content": code
            }
        ]
    )

    raw_text = answer.message.content[0].text.strip()

    # strip markdown backticks if Cohere wraps the JSON
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
        result = CodeReview(**data)
        return result.model_dump()
    except Exception as e:
        return {"error": str(e)}

# ── Agent setup ───────────────────────────────────────────────────────────────
co = cohere.ClientV2(api_key)
user_message = input("You: ")

# conversation history — grows with every step of the loop
messages = [{"role": "user", "content": user_message}]

# ── Agent loop ────────────────────────────────────────────────────────────────
iterations = 0

while iterations < 10:
    # call the LLM with the full conversation history and tool definitions
    response = co.chat(
        model="command-r-plus-08-2024",
        messages=messages,
        tools=tools
    )

    if response.message.tool_calls:
        # append the assistant's decision to history
        # (Cohere needs this before the tool results)
        messages.append(response.message)

        # handle every tool call in this response
        # (Cohere can request multiple tools at once)
        for tool_call in response.message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"→ [{iterations + 1}] Calling: {tool_name}({tool_args})")

            # execute the right function
            if tool_name == "review_code":
                result = review_code(tool_args["code"])
            elif tool_name == "calculate":
                result = calculate(tool_args["expression"])
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            # append tool result to history
            # tool_call_id links this result to the specific tool call
            messages.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })

        iterations += 1

    else:
        # no tool calls — LLM is done, print final answer and exit loop
        print(f"\n{response.message.content[0].text}")
        break

else:
    # while loop exhausted 10 iterations without a final answer
    print("\n[Agent stopped — max iterations reached]")