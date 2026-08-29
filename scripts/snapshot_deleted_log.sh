#!/usr/bin/env bash
# 把一个仍在运行的进程持有的**已删除**日志文件周期性快照回磁盘。
#
#   ./scripts/snapshot_deleted_log.sh <PID> <目标目录> [间隔秒]
#
# 文件被 rm 掉之后, 只要还有进程开着它, inode 就不会释放, 内容可以从
# /proc/<PID>/fd/<N> 原样读出来。但**进程一退出 inode 就真没了** —— 所以要么现在
# 就 cp 一份, 要么像这里一样定期快照。没有系统调用能把它重新 link 回目录树。
set -uo pipefail
PID=${1:?用法: $0 <PID> <目标目录> [间隔秒]}
DST=${2:?用法: $0 <PID> <目标目录> [间隔秒]}
EVERY=${3:-60}
mkdir -p "$DST"

snap() {
    local n t base
    for n in /proc/$PID/fd/*; do
        t=$(readlink "$n" 2>/dev/null) || continue
        [[ "$t" == *"(deleted)"* ]] || continue
        base=$(basename "${t% (deleted)}")
        [[ -n "$base" ]] && cp "$n" "$DST/$base" 2>/dev/null
    done
}

echo "[snapshot] pid=$PID -> $DST  每 ${EVERY}s 一次"
while kill -0 "$PID" 2>/dev/null; do
    snap
    sleep "$EVERY"
done
snap                                  # 进程刚退出时补最后一次 (fd 可能已释放)
echo "[snapshot] pid=$PID 已退出, 最后一次快照完成 $(date '+%F %T')"
