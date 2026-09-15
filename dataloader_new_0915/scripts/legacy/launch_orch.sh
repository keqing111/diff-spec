#!/bin/bash
# Launch the diagnostic orchestrator detached, avoiding pkill self-match issues.
D=/home/y50063564/dataloader_diag
mkdir -p "$D/logs"
running=$(ps -ef | grep -c "[r]un_diag_sequence.sh")
if [ "$running" -gt 0 ]; then
  echo "orchestrator already running ($running) -> not starting another"
  exit 0
fi
nohup bash "$D/run_diag_sequence.sh" > "$D/logs/orchestrator.out" 2>&1 &
echo "orchestrator started pid $!"
