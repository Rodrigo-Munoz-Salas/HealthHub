# Run the UI (React/Vite)
run-ui:
	cd ui && npm run dev

# Run the Go API
run-api:
	cd internal/cmd/api && go run .

# Run the Python Agent
run-agent:
	cd agent && . .venv/bin/activate && uvicorn src.main:app --host 0.0.0.0 --port 8001

# Run everything together
run-all:
	concurrently "make run-ui" "make run-api" "make run-agent"
