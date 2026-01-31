#!/bin/bash

# Start servers for end-to-end testing
# Usage: ./start_servers.sh

set -e

echo "Starting Magi validation servers..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please copy .env.example to .env and set your OPENROUTER_API_KEY"
    exit 1
fi

# Check if OPENROUTER_API_KEY is set
source .env
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY not set in .env file"
    exit 1
fi

# Create log directory
mkdir -p logs

# Start NLI server
echo "[1/2] Starting NLI server on port 9000..."
python nli_openrouter_server.py > logs/nli_server.log 2>&1 &
NLI_PID=$!
echo "  NLI server started (PID: $NLI_PID)"

# Wait a bit for the server to start
sleep 2

# Start LettuceDetect server
echo "[2/2] Starting LettuceDetect server on port 8000..."
python lettuce_server.py > logs/lettuce_server.log 2>&1 &
LETTUCE_PID=$!
echo "  LettuceDetect server started (PID: $LETTUCE_PID)"

# Save PIDs to file for easy cleanup
echo "$NLI_PID" > logs/nli_server.pid
echo "$LETTUCE_PID" > logs/lettuce_server.pid

echo ""
echo "Both servers are running!"
echo ""
echo "Logs:"
echo "  NLI:          tail -f logs/nli_server.log"
echo "  LettuceDetect: tail -f logs/lettuce_server.log"
echo ""
echo "To run tests:"
echo "  python test_end_to_end.py"
echo ""
echo "To stop servers:"
echo "  ./stop_servers.sh"
echo ""
