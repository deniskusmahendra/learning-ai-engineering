"""
The backend: a small FastAPI service that analyzes ONE customer review.

Run it with:
    uv run fastapi dev api.py

Then the Streamlit app (app.py) will call this service for every review.
"""

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)
client = Groq()

app = FastAPI()


# What the caller must SEND us
class Review(BaseModel):
    model_config=ConfigDict(extra="forbid")
    text: str


# What Groq gives back, and what we SEND to the caller.
# We keep it small on purpose -> fewer tokens used.
class Analysis(BaseModel):
    model_config=ConfigDict(extra="forbid")
    label: str   # "positive", "negative", or "neutral"
    score: int   # 1 (very bad) to 5 (very good)
    theme: str   # one word: what the review is mainly about (e.g. "delivery")


prompt = """Analyze this customer review.\n
label must be 'positive', 'negative', or 'neutral'.\n
score must be a number from 1 (very bad) to 5 (very good).\n
theme must be ONE lowercase word for the main topic (for example: delivery, taste, price, service, quality).\n"""


@app.post("/analyze")
def analyze(review: Review):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "system", "content": f"{prompt}Review: {review.text}"}],
        response_format={
           "type": "json_schema",
           "json_schema": {
               "name": "feedback_analyzer",
               "strict": True,
               "schema": Analysis.model_json_schema()
            }
        }
    )
    return Analysis.model_validate_json(response.choices[0].message.content)
