from dotenv import load_dotenv
import cohere
import os
import json
from pydantic import BaseModel, model_validator

class CodeReview(BaseModel):
    score: int
    issues: list[str]
    suggestion: str
    approved: bool = False

    @model_validator(mode='after')
    def set_approved(self):
        self.approved = self.score >= 7
        return self

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

co = cohere.ClientV2(api_key)


query = input("Enter your code to review: ")

answer = co.chat(
    model="command-r-plus-08-2024",
    messages=[
        {
            "role": "system",
            "content": """ You are a code reviewer , your main job is to accept code and review it and then return the review only as a valid JSON with this structure :
{
  "score": ...,
  "issues": [...],
  "suggestion": "...",
  "approved": ...
}"""
        },
        {
            "role": "user",
            "content": query
        }
    ]
)

raw_text = answer.message.content[0].text
raw_text = raw_text.strip()
if raw_text.startswith("```"):
    raw_text = raw_text.split("```")[1]
    if raw_text.startswith("json"):
        raw_text = raw_text[4:]
raw_text = raw_text.strip()

try:
    data = json.loads(raw_text)
    result = CodeReview(**data)
    print(f"\nScore:      {result.score}/10")
    print(f"Approved:   {result.approved}")
    print(f"Issues:     {result.issues}")
    print(f"Suggestion: {result.suggestion}")
except json.JSONDecodeError:
    print("Error: LLM returned invalid JSON")
except Exception as e:
    print(f"Validation error: {e}")
