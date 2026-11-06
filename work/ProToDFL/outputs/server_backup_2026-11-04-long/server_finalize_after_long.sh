#!/bin/bash
set -u
cd /home/seczone/ysz
mkdir -p outputs/console
LOG=outputs/console/finalize-after-long.log
: > "$LOG"

echo "waiting for srv20-e5-r10 runs" >> "$LOG"
for i in $(seq 1 120); do
  count=$(ps -ef | grep '[m]ain_ProToDFL.*srv20-e5-r10' | grep -v grep | wc -l)
  if [ "$count" -eq 0 ]; then
    echo "training finished at iteration $i" >> "$LOG"
    break
  fi
  if [ "$i" -eq 120 ]; then
    echo "TIMEOUT: training still running" >> "$LOG"
    exit 1
  fi
  sleep 30
done

echo "running final paired checks" >> "$LOG"
bash scripts/server_final_paired.sh > outputs/console/final-paired.log 2>&1
echo "final paired checks done rc=$?" >> "$LOG"

rm -rf outputs_backup_2026-11-04-long
mkdir -p outputs_backup_2026-11-04-long
cp -r outputs/logs outputs/checks outputs/console outputs_backup_2026-11-04-long/
find outputs/checkpoints -name metadata.json -exec cp --parents {} outputs_backup_2026-11-04-long/ \;
cp scripts/run_server_long_pairs.sh scripts/server_final_paired.sh scripts/server_finalize_after_long.sh requirements.txt outputs_backup_2026-11-04-long/
tar -czf outputs_backup_2026-11-04-long.tar.gz -C outputs_backup_2026-11-04-long .
rm -rf outputs_backup_2026-11-04-long outputs_backup_2026-11-04 outputs_backup_2026-11-04.tar.gz protodfl_code.tar.gz

echo "backup size" >> "$LOG"
du -sh outputs_backup_2026-11-04-long.tar.gz >> "$LOG"
echo "gpu" >> "$LOG"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$LOG"
echo "disk" >> "$LOG"
df -h . | tail -1 >> "$LOG"
echo "DONE" > outputs/console/FINALIZE_DONE
