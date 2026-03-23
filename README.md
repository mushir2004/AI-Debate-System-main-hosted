# 🎙️ DebateForge: Autonomous Multi-Agent AI Debate System

![DebateForge Logo](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![Ollama](https://img.shields.io/badge/AI-Ollama_Local-orange)

**DebateForge** is a stateful, multi-agent AI debate platform designed to act as an aggressive, real-time sparring partner for Group Discussion (GD) and interview preparation. 

![DebateForge Home Screen](https://github.com/prajjwal-17/AI-Debate-System/blob/main/assests/home.jpg)

## 📖 The Origin Story
Standard AI chatbots are built for subservience—they generate walls of text when you ask them for topics. But reading a list of bullet points doesn't prepare you for the real-time pressure, interruptions, and logical clashes of a high-stakes Group Discussion. 

I didn't need a text generator. I needed a relentless, vocal sparring partner to test my logic on the fly. I needed someone to argue with me and call out my bad logic. That didn't exist... so I built it.

## 🏗️ System Architecture

```mermaid
graph TD
    User([Human User]) -->|Interacts| UI[Frontend Interface]
    UI -->|REST API| API[FastAPI Backend]

    subgraph "DebateForge Orchestration Engine"
        API <-->|Manages| Context[DebateSession Memory]
        API -->|Audio Gen| WebSpeech[Web Speech API / gTTS]
        API -->|Prompt + Context| LocalAI[(Ollama: Llama 3 / Qwen)]
        LocalAI -->|Inference JSON| API
    end

    subgraph "Human-in-the-Loop Pipeline"
        API -->|Raw Text| Fallacy[Fallacy Detector]
        Fallacy -->|Identifies Logic Flaws| Steelman[Steelman Refactor]
        Steelman -->|Injects Strong Argument| Context
    end

    API -->|JSON + Audio MP3| UI
    
    subgraph "Gamification"
        LocalAI -->|LLM-as-a-Judge| Leaderboard[(leaderboard.json)]
    end
```

## ✨ Core Features

### 1. Autonomous Multi-Agent Debates
Set a topic, pick your fighters, and watch two AI models debate complex subjects autonomously in real-time.

![Debate Setup Configuration](https://github.com/prajjwal-17/AI-Debate-System/blob/main/assests/setup.jpeg)

### 2. Dynamic Personas
System prompting forces the AI into specific psychological profiles that never break character. 
* 🎓 **The Professor:** Highly analytical, elite academic tone.
* 🧌 **The Troll:** Dismissive, slang-heavy, intentionally frustrating but logically sound.
* 🔥 **The Aggressor & 🏛️ The Philosopher.**

![Live Debate Arena](https://github.com/prajjwal-17/AI-Debate-System/blob/main/assests/arena.jpeg)

### 3. Real-Time NLP Interruption (The Sandbox)
Humans can pause the autonomous loop at any time and type their own arguments to attack either persona. 
* **Fallacy Detection:** The engine instantly flags logical flaws (e.g., False Dichotomy, Ad Hominem) in the human's input.
* **Steelman Protocol:** Strips away human emotion and rewrites the argument into its strongest logical form before injecting it into the debate state.

![Human Interrupt and Steelman feature](https://github.com/prajjwal-17/AI-Debate-System/blob/main/assests/interrupt.jpg)

### 4. LLM-as-a-Judge & Multilingual Support
An impartial AI evaluates the full transcript, declares a winner based on logical merit, and updates a persistent leaderboard. The platform also features multilingual support (e.g., Hindi) for broader accessibility.

![Judgment and Multilingual Verdict](https://github.com/prajjwal-17/AI-Debate-System/blob/main/assests/verdict.jpeg)

## 🛠️ Tech Stack

* **Frontend:** NextJS, Tailwind CSS) & Framer Motion
* **Backend:** Python, FastAPI
* **AI/Inference:** Ollama (Running local Llama 3)
* **Audio:** Web Speech API / `gTTS`

## 🚀 Local Setup & Installation

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed on your machine.
* At least 8GB of RAM (A dedicated GPU is recommended for faster token generation).

### 2. Start the AI Engine
Open a terminal and pull your preferred model. Leave this running in the background to keep the model warm in VRAM.
```bash
# We recommend Llama 3 (9B)
ollama run llama3

# We recommend Microsoft Edge
```

### 3. Backend Setup
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/prajjwal-17/AI-Debate-System.git](https://github.com/prajjwal-17/AI-Debate-System.git)
cd AI-Debate-System/backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install requirements
pip install fastapi uvicorn pydantic gtts

# Run the server
uvicorn main:app --reload
```
The backend will now be running at `http://localhost:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

### 4. Frontend Setup
```bash
cd ../frontend
# Add your specific frontend run commands here
```

## 🔮 Future Roadmap
- [ ] **Zero-Latency Look-Ahead Buffer:** Implement FastAPI background tasks to pre-compute AI responses asynchronously while audio plays.
- [ ] **Expanded Indic Multilingual Support:** Scale the dynamic prompts to allow agents to debate natively in Bengali, Tamil, and Telugu.

---
*Built to win arguments. Engineered to run locally.*
