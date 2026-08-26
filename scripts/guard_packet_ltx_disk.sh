#!/usr/bin/env bash
set -euo pipefail

# Packet advertises a 150 GB ephemeral volume, which is ~134 GiB in the
# container.  A mixed LongCat + LTX 2.5 batch can briefly require more than
# that if both downloaders run unrestricted.  Let both begin in parallel, then
# reserve room for LongCat as soon as the shared root approaches the floor.

LTX_PID_FILE="${LTX_PID_FILE:?LTX_PID_FILE is required}"
LONGCAT_WEIGHTS_DIR="${LONGCAT_WEIGHTS_DIR:?LONGCAT_WEIGHTS_DIR is required}"
DISK_PATH="${PACKET_DISK_GUARD_PATH:-/workspace}"
MIN_FREE_GB="${AI_VIDEO_GEN_PACKET_LTX_MIN_FREE_GB:-70}"
POLL_SEC="${AI_VIDEO_GEN_PACKET_LTX_GUARD_POLL_SEC:-5}"

ltx_pid="$(cat "${LTX_PID_FILE}" 2>/dev/null || true)"
if ! [[ "${ltx_pid}" =~ ^[0-9]+$ ]]; then
  echo "Packet LTX disk guard: downloader PID is unavailable; nothing to guard." >&2
  exit 0
fi

seen_longcat_weights=0
paused=0

free_gib() {
  df -Pk "${DISK_PATH}" | awk 'NR == 2 { printf "%d", $4 / 1024 / 1024 }'
}

while kill -0 "${ltx_pid}" 2>/dev/null; do
  if [ -d "${LONGCAT_WEIGHTS_DIR}" ]; then
    seen_longcat_weights=1
  fi

  free="$(free_gib)"
  if [ "${seen_longcat_weights}" = "1" ] \
    && [ -d "${LONGCAT_WEIGHTS_DIR}" ] \
    && [ "${paused}" = "0" ] \
    && [ "${free}" -lt "${MIN_FREE_GB}" ]; then
    kill -STOP "${ltx_pid}"
    paused=1
    echo "Packet LTX disk guard: paused LTX downloader at ${free} GiB free; waiting for LongCat branch cleanup." >&2
  fi

  if [ "${paused}" = "1" ] && [ ! -d "${LONGCAT_WEIGHTS_DIR}" ]; then
    kill -CONT "${ltx_pid}"
    echo "Packet LTX disk guard: LongCat weights released; resumed LTX downloader." >&2
    exit 0
  fi

  sleep "${POLL_SEC}"
done
