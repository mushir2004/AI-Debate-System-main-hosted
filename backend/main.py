from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uuid
import json

from ai_engine import detect_fallacies, steelman_argument, judge_debate
from moderator import start_debate, generate_next_turn, active_sessions

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="DebateForge Autonomous API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartRequest(BaseModel):
    topic: str
    pro_persona: str = "The Professor"
    con_persona: str = "The Aggressor"
    language: str    = "english"          # ← new field, defaults to english

class NextRequest(BaseModel):
    session_id: str

class InterruptRequest(BaseModel):
    session_id: str
    user_argument: str
    target_persona: str

@app.post("/api/debate/start")
async def api_start_debate(request: StartRequest):
    session_id = str(uuid.uuid4())
    first_turn = start_debate(
        session_id,
        request.topic,
        request.pro_persona,
        request.con_persona,
        request.language,              # ← pass language to moderator
    )
    return {
        "session_id":    session_id,
        "topic":         request.topic,
        "pro_character": request.pro_persona,
        "con_character": request.con_persona,
        "language":      request.language,
        "first_turn":    first_turn,
    }

@app.post("/api/debate/next")
async def api_next_turn(request: NextRequest):
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Debate session not found.")
    return await generate_next_turn(request.session_id)

@app.post("/api/debate/interrupt")
async def api_interrupt(request: InterruptRequest):
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Debate session not found.")
    session = active_sessions[request.session_id]
    fallacies        = detect_fallacies(request.user_argument)
    steelmanned_text = steelman_argument(request.user_argument)
    session.add_message("User", request.user_argument)
    session.current_turn = request.target_persona
    return {
        "status":             "Interrupted Successfully",
        "fallacies_detected": fallacies,
        "steelmanned_version": steelmanned_text,
        "next_speaker":       session.current_turn,
    }

@app.get("/api/personas/stats")
async def get_leaderboard():
    with open("leaderboard.json", "r") as f:
        return json.load(f)

@app.post("/api/debate/judge")
async def api_judge_debate(request: NextRequest):
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    session    = active_sessions[request.session_id]
    transcript = "\n".join([
        f"{msg['speaker']} ({session.characters.get(msg['speaker'], 'User')}): {msg['text']}"
        for msg in session.history
    ])
    result = judge_debate(transcript)
    if result["winner"] in ["Pro", "Con"]:
        winning_character = session.characters[result["winner"]]
        with open("leaderboard.json", "r") as f:
            leaderboard = json.load(f)
        if winning_character in leaderboard:
            leaderboard[winning_character]["wins"] += 1
            leaderboard[winning_character]["last_brag"] = result["brag"]
        with open("leaderboard.json", "w") as f:
            json.dump(leaderboard, f)
    return {
        "verdict":           result,
        "winning_character": session.characters.get(result.get("winner"), "None"),
    }