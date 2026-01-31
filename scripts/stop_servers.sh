#!/bin/bash

# Stop servers started by start_servers.sh
# Usage: ./stop_servers.sh

echo "Stopping Magi validation servers..."

if [ -f logs/nli_server.pid ]; then
    NLI_PID=$(cat logs/nli_server.pid)
    if kill -0 "$NLI_PID" 2>/dev/null; then
        kill "$NLI_PID"
        echo "  Stopped NLI server (PID: $NLI_PID)"
    else
        echo "  NLI server not running"
    fi
    rm logs/nli_server.pid
else
    echo "  NLI server PID file not found"
fi

if [ -f logs/lettuce_server.pid ]; then
    LETTUCE_PID=$(cat logs/lettuce_server.pid)
    if kill -0 "$LETTUCE_PID" 2>/dev/null; then
        kill "$LETTUCE_PID"
        echo "  Stopped LettuceDetect server (PID: $LETTUCE_PID)"
    else
        echo "  LettuceDetect server not running"
    fi
    rm logs/lettuce_server.pid
else
    echo "  LettuceDetect server PID file not found"
fi

echo ""
echo "All servers stopped."
