#!/usr/bin/env bash
set -Eeuo pipefail
cat >&2 <<'MSG'
OpenVLA 的本地评测代码目前只提供标准 LIBERO（见 eval_openvla_libero.sh）；
当前归档没有可用的 LIBERO-plus 评测入口。
MSG
exit 2

