#!/usr/bin/env bash
# PurePacketPPSeq Model Explorer launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PORT=${1:-8050}

echo "==========================================================================="
echo " Starting PurePacketPPSeq Interactive Model Explorer"
echo " Web URL:  http://localhost:${PORT}"
echo " Local Runs Folder: $DIR/runs"
echo "==========================================================================="

exec python3 "$DIR/explorer/server.py" --port "$PORT"
