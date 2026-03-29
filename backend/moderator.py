import json
import os
import asyncio
from typing import List, Dict
from ai_engine import query_model, generate_rebuttal

active_sessions  = {}
LEADERBOARD_FILE = "leaderboard.json"

# ── The 4 Distinct Personas ───────────────────────────────────────────────────
PERSONAS = {
    "The Professor": "You are highly analytical, use big vocabulary, and speak like an Oxford academic.",
    "The Aggressor":  "You are ruthless, fiery, and use sharp, biting sarcasm to mock your opponent.",
    "The Philosopher":"You are calm, existential, and constantly question the deeper meaning of the topic.",
    "The Troll":      "You are dismissive, use internet slang, and try to annoy your opponent while still making logical points.",
}

# ── Language instructions injected into every system prompt ──────────────────
LANGUAGE_INSTRUCTIONS = {
    "english": "",
    "hindi":   "IMPORTANT: You MUST respond ONLY in Hindi using Devanagari script. Do not use any English words except proper nouns.",
    "bengali": "IMPORTANT: You MUST respond ONLY in Bengali using Bengali script. Do not use any English words except proper nouns.",
    "tamil":   "IMPORTANT: You MUST respond ONLY in Tamil using Tamil script. Do not use any English words except proper nouns.",
    "telugu":  "IMPORTANT: You MUST respond ONLY in Telugu using Telugu script. Do not use any English words except proper nouns.",
}

# Initialize leaderboard if missing
if not os.path.exists(LEADERBOARD_FILE):
    initial_data = {
        name: {"wins": 0, "last_brag": "I'm ready for my first victim."}
        for name in PERSONAS.keys()
    }
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(initial_data, f)


class DebateSession:
    def __init__(
        self,
        session_id: str,
        topic:      str,
        pro_persona: str = "The Professor",
        con_persona: str = "The Aggressor",
        language:    str = "english",          # ← new param
    ):
        self.session_id   = session_id
        self.topic        = topic
        self.language     = language.lower()   # ← stored on session
        self.history: List[Dict[str, str]] = []
        self.is_paused    = False
        self.current_turn = "Pro"
        self.turn_lock    = asyncio.Lock()

        self.characters = {
            "Pro": pro_persona if pro_persona in PERSONAS else "The Professor",
            "Con": con_persona if con_persona in PERSONAS else "The Aggressor",
        }

    def get_llm_messages(self, stance: str) -> List[Dict[str, str]]:
        character_name   = self.characters[stance]
        personality      = PERSONAS[character_name]
        lang_instruction = LANGUAGE_INSTRUCTIONS.get(self.language, "")

        system_msg = (
            f"You are {character_name}. {personality} "
            f"The topic is '{self.topic}'. "
            f"Your strict stance is {stance}. "
            f"Review the debate history and provide a concise, 1-paragraph argument. "
            f"Directly attack the opponent's flaws in your unique voice. "
            f"{lang_instruction}"   # ← injected at end of system prompt
        )

        messages = [{"role": "system", "content": system_msg}]
        for entry in self.history:
            role = "assistant" if entry["speaker"] == stance else "user"
            messages.append({"role": role, "content": entry["text"]})

        return messages

    def add_message(self, speaker: str, text: str):
        self.history.append({"speaker": speaker, "text": text})


def start_debate(
    session_id:  str,
    topic:       str,
    pro_persona: str,
    con_persona: str,
    language:    str = "english",          # ← new param
):
    """Initializes a new debate and generates the opening statement."""
    session = DebateSession(session_id, topic, pro_persona, con_persona, language)
    active_sessions[session_id] = session

    messages = session.get_llm_messages("Pro")
    response = query_model(messages)

    session.add_message("Pro", response)
    session.current_turn = "Con"

    return {"speaker": "Pro", "text": response, "history": session.history}


async def generate_next_turn(session_id: str):
    """Triggers the next AI to speak. Turn lock prevents concurrent prefetch
    requests from generating duplicate or out-of-order turns."""
    session = active_sessions.get(session_id)

    if not session:
        return {"error": "Session not found."}

    if session.is_paused:
        return {"status": "paused", "message": "Debate is paused for user intervention."}

    async with session.turn_lock:
        stance   = session.current_turn
        messages = session.get_llm_messages(stance)

        loop     = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, query_model, messages)

        session.add_message(stance, response)
        session.current_turn = "Con" if stance == "Pro" else "Pro"

    return {"speaker": stance, "text": response, "history": session.history}