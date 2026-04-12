from dotenv import load_dotenv
import cohere
import os
import json
from pydantic import BaseModel, model_validator


load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

class CodeReview(BaseModel):
    score: int
    issues: list[str]
    suggestion: str
    approved: bool = False

    @model_validator(mode='after')
    def set_approved(self):
        self.approved = self.score >= 7
        return self

tools = [
    {
        "type": "function",
        "function": {
            "name": "review_code",
            "description": "Reviews code and returns a score, issues, and suggestion",
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
            "description": "Evaluates a math expression and returns the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. 847 * 293"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


def calculate(expression: str) -> str:
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
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
        result = CodeReview(**data)
        return result.model_dump()  # return as dict
    except Exception as e:
        return {"error": str(e)}

co = cohere.ClientV2(api_key)
user_message = input("You: ")

messages = [{"role": "user", "content": user_message}]

# First call — let Cohere decide which tool to use
response = co.chat(
    model="command-r-plus-08-2024",
    messages=messages,
    tools=tools
)

# Check if Cohere wants to call a tool
if response.message.tool_calls:
    tool_call = response.message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    print(f"\n→ Calling tool: {tool_name}")
    print(f"→ With args: {tool_args}")

    # Execute the right function
    if tool_name == "review_code":
        result = review_code(tool_args["code"])
    elif tool_name == "calculate":
        result = calculate(tool_args["expression"])
    else:
        result = {"error": "Unknown tool"}

    print(f"\nResult: {result}")

else:
    # No tool needed — just a regular answer
    print(f"\n{response.message.content[0].text}")