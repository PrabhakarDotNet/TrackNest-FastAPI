from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ── App ───────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "https://tracknest-fastapi-gnhjcafkayf9hwfy.westcentralus-01.azurewebsites.net",
        "https://tracknest-api-grbfe3ascsdsh6c0.malaysiawest-01.azurewebsites.net",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Clients ───────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# ── In-memory session store ───────────────────────────
sessions: dict[str, list] = {}

def get_session_history(session_id: str) -> list:
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]

# ── Models ────────────────────────────────────────────
class DescriptionRequest(BaseModel):
    description: str

class CategoryResponse(BaseModel):
    category: str

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    expenses: list[dict] = []

class ChatResponse(BaseModel):
    reply: str
    session_id: str

# ── Categories ────────────────────────────────────────
CATEGORIES = [
    "Food & Dining", "Transport", "Shopping", "Health",
    "Bills & Utilities", "Entertainment", "Sports & Fitness",
    "Education", "Investment", "Other"
]

# ── Suggest Category ──────────────────────────────────
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

# ── Chat ──────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    history = get_session_history(request.session_id)

    # Build expense context
    if request.expenses:
        expense_lines = "\n".join(
            f"- {e.get('description','N/A')}: ₹{e.get('amount', 0)} "
            f"({e.get('category','N/A')}) on {e.get('expenseDate','N/A')}"
            for e in request.expenses
        )
        expense_context = f"User's actual expenses ({len(request.expenses)} records):\n{expense_lines}"
    else:
        expense_context = "No expense data was provided for this session."

    system_prompt = f"""You are TrackNest AI, a personal expense tracking assistant.

STRICT RULES:
- Only answer questions about the user's expenses, spending habits, budgeting, and personal finance.
- NEVER invent, guess, or hallucinate expense data. Only use the data provided below.
- If no expense data is provided, clearly say: "I don't see any expenses loaded yet. Please add some expenses in TrackNest first."
- If asked about login credentials, accounts, or anything outside finance — politely decline.
- Do not suggest the user "link their account" — they are already logged in and their data is shown below.
- Always respond in a concise, friendly tone.

{expense_context}"""

    messages = [("system", system_prompt)]

    for msg in history:
        messages.append((msg["role"], msg["content"]))

    messages.append(("human", request.message))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm
    response = chain.invoke({})
    reply = response.content

    history.append({"role": "human", "content": request.message})
    history.append({"role": "assistant", "content": reply})

    return ChatResponse(reply=reply, session_id=request.session_id)

# ── Clear Session ─────────────────────────────────────
@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session cleared"}

# ── Health Check ──────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}
