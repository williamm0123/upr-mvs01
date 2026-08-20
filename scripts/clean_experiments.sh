#!/bin/bash
# 清空 log/experiments/ —— 历次消融 arm 的 checkpoint、tensorboard 和归档。
#
#     bash scripts/clean_experiments.sh          # 只列出会删什么
#     bash scripts/clean_experiments.sh --yes    # 真的删
#
# 删除不可逆。会一起消失的东西:
#   * 各 arm 的 best.pth —— 点云 eval (scripts/test_dtu.sh) 的输入
#   * L 那份 NaN 的 latest.pth —— 发散事故的取证材料
#   * 全部 tensorboard 事件文件 —— compare_arms.py 唯一的数据源
# 想留证据就先把 log/experiments/_archive 拷走再跑这个。
#
# 注意: 各 arm 脚本自己会在开跑前把同名 run 移到 _archive/, 所以"重跑一轮"
# 本来就不需要先清。这个脚本是给"从头再来、连归档也不要了"用的。
set -e
cd "$(dirname "$0")/.."

ROOT="log/experiments"
if [[ ! -d "$ROOT" ]]; then
    echo "$ROOT 不存在, 无需清理"
    exit 0
fi

echo "=== 将被删除 ==="
du -sh "$ROOT"/* 2>/dev/null || echo "  (空)"
TOTAL=$(du -sh "$ROOT" 2>/dev/null | cut -f1)
echo "=== 合计 $TOTAL ==="

if [[ "${1:-}" != "--yes" ]]; then
    echo
    echo "这是预演, 什么都没删。确认无误后加 --yes:"
    echo "    bash scripts/clean_experiments.sh --yes"
    exit 0
fi

rm -rf "${ROOT:?}"/*
echo "已清空 $ROOT"
