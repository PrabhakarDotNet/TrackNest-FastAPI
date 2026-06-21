from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

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

# ── Clients ──────────────────────────────────────────
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


# ── Existing endpoint (unchanged) ─────────────────────
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


# ── New chat endpoint ──────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    history = get_session_history(request.session_id)

    # Build expense context
    expense_context = ""
    if request.expenses:
        expense_context = "User's current expenses:\n"
        for e in request.expenses:
            expense_context += f"- {e.get('description','N/A')}: ₹{e.get('amount',0)} ({e.get('category','N/A')})\n"

    # Build messages for LangChain
    messages = [
        ("system", f"""You are TrackNest AI, a helpful personal expense tracking assistant.
Help users understand their spending habits and answer questions about expenses.
Be concise, friendly and helpful.
{expense_context}"""),
    ]

    # Add conversation history
    for msg in history:
        messages.append((msg["role"], msg["content"]))

    # Add current user message
    messages.append(("human", request.message))

    # Call LLM
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm
    response = chain.invoke({})
    reply = response.content

    # Save to session history
    history.append({"role": "human", "content": request.message})
    history.append({"role": "assistant", "content": reply})

    return ChatResponse(reply=reply, session_id=request.session_id)


@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session cleared"}


# ── Health check ───────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}