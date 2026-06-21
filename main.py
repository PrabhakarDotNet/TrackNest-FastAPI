from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class DescriptionRequest(BaseModel):
    description: str

class CategoryResponse(BaseModel):
    category: str

CATEGORIES = [
    "Food & Dining",
    "Transport",
    "Shopping",
    "Health",
    "Bills & Utilities",
    "Entertainment",
    "Sports & Fitness",
    "Education",
    "Investment",
    "Other"
]

@app.post("/suggest-category", response_model=CategoryResponse)
async def suggest_category(request: DescriptionRequest):
    prompt = f"""
You are an expense categorization assistant.
Given a description, return ONLY one category from this list:
{", ".join(CATEGORIES)}

Description: "{request.description}"

Reply with ONLY the category name, nothing else.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0
    )

    category = response.choices[0].message.content.strip()

    if category not in CATEGORIES:
        category = "Other"

    return CategoryResponse(category=category)

@app.get("/health")
async def health():
    return {"status": "ok"}