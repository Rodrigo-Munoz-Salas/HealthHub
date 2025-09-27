# HealthHub System

This project is a 3-service system built with **React (Vite)**, **Go (REST API)**, and **Python (FastAPI)**.  

- **UI** (Vite + React): Dashboard where users can create an account and chat with the AI.  
- **Go API**: Provides REST endpoints for creating users and verifying them (UUID based).  
- **Python Agent**: AI agent stub (currently just echoes input). Later can be extended to call Google GenAI.  

---

## Prerequisites
Make sure you have these installed:

- [Node.js](https://nodejs.org/) (≥ 18.x recommended)  
- [Go](https://go.dev/) (≥ 1.22)  
- [Python 3](https://www.python.org/) (≥ 3.10)  
- [Make](https://www.gnu.org/software/make/) (comes with Linux/macOS; for Windows use Git Bash or WSL)  
- `concurrently` (Node helper for running multiple processes at once)  

Install `concurrently` globally once:
```
npm install -g concurrently
```

## Setup the UI
```
cd ui
npm install
```
Create a ```.env``` file in ```ui/```:
```
VITE_API_URL=http://localhost:8080
VITE_AGENT_URL=http://localhost:8001
```

## Setup the GO API
```
cd internal
go mod tidy
```

## Setup the Python Agent
```
cd agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
then
```
deactivate
```

Create a ```.env``` file in ```agent/``` for implementing agent:
```
GEMINI_API_KEY=
```

## Run everything
```
make run-all
```