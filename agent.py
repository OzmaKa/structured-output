from dotenv import load_dotenv
import cohere
import os
import json
import ast
import operator
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

# ── Tool definitions ───────────────────────────────────────────────────────────
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

# ── Safe math evaluator (no eval) ─────────────────────────────────────────────
ALLOWED_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
}

def safe_eval(expr: str):
    tree = ast.parse(expr, mode='eval')
    return _eval(tree.body)

def _eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op = ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError("Operation not allowed")
        return op(_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = ALLOWED_OPS.get(type(node.op))
        return op(_eval(node.operand))
    else:
        raise ValueError(f"Expression not allowed: {type(node)}")

# ── Tool implementations ───────────────────────────────────────────────────────
def calculate(expression: str) -> str:
    expression = expression.replace("^", "**")
    result = safe_eval(expression)
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
        return result.model_dump()
    except Exception as e:
        return {"error": str(e)}

# ── Agent ─────────────────────────────────────────────────────────────────────
co = cohere.ClientV2(api_key)

print("Agent ready. Type 'exit' to quit.\n")

# ── Outer loop — keeps the agent alive between inputs ─────────────────────────
while True:
    user_message = input("You: ")

    # exit condition
    if user_message.strip().lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # skip empty input
    if not user_message.strip():
        continue

    # fresh conversation for each new input
    messages = [{"role": "user", "content": user_message}]
    iterations = 0

    # ── Inner loop — agent runs until done or max iterations ──────────────────
    while iterations < 10:
        response = co.chat(
            model="command-r-plus-08-2024",
            messages=messages,
            tools=tools
        )

        if response.message.tool_calls:
            messages.append(response.message)

            for tool_call in response.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"→ [{iterations + 1}] Calling: {tool_name}({tool_args})")

                if tool_name == "review_code":
                    result = review_code(tool_args["code"])
                elif tool_name == "calculate":
                    result = calculate(tool_args["expression"])
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id
                })

            iterations += 1

        else:
            print(f"\nAgent: {response.message.content[0].text}\n")
            break

    else:
        print("\n[Agent stopped — max iterations reached]\n")