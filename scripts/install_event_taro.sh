#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
TARO_ROOT="/root/autodl-tmp/taro"
VERL_ROOT="${TARO_ROOT}/verl"

install -m 0644 "${SRC_DIR}/taro_event_manager.py" \
  "${VERL_ROOT}/verl/experimental/agent_loop/taro_event_manager.py"
install -m 0644 "${SRC_DIR}/taro_event_online.py" \
  "${VERL_ROOT}/verl/trainer/ppo/taro_event_online.py"

python "${SRC_DIR}/switch_to_event_taro.py"

cd "${VERL_ROOT}"
python -m py_compile \
  verl/experimental/agent_loop/taro_event_manager.py \
  verl/trainer/ppo/taro_event_online.py \
  verl/trainer/ppo/ray_trainer.py

python - <<'PY'
from verl.experimental.agent_loop.taro_event_manager import TAROEventAgentLoopManager
from verl.trainer.ppo.taro_event_online import taro_generate_step
print('Event-TARO import: PASS')
PY
