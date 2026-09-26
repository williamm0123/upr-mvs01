#!/bin/bash
# =============================================================================
# BlendedMVS 低分辨率版 (768x576, 113 场景) 一条龙: 等下载完 -> 解压 -> 拍平 -> 校验 -> 删 zip -> 上传。
#
#   tmux new -s blended
#   bash scripts/prepare_blendedmvs_lowres.sh 2>&1 | tee ~/prepare_blended_lowres.log
#
# 开头就会连一次 umhpc 让你输密码 (连接会一直保持), 之后下载/解压/上传都不用再管。
# 每一步都可重跑: 已解压且校验通过就跳过解压; 上传是 rsync 断点续传。
#
# 流程:
#   1. 等待:   $DOWNLOADS 里没有 *.crdownload / *.part, 且出现 BlendedMVS 的 zip
#              (默认匹配 BlendedMVS.zip / BlendedMVS.z01.. / dataset_low_res*.zip;
#               不认 BlendedMVS1/2 —— 那是 BlendedMVS+ / ++)。也可 ZIPS="a.zip b.zip" 显式指定
#   2. 测试:   unzip -t 全量校验压缩包 (分卷先用 zip -s 0 合并)
#   3. 解压:   到 $TARGET 同盘的临时目录, 再把每个场景 (blended_images/ 的父目录, 不管嵌套几层)
#              mv 成 $TARGET/<PID>/, 空的 occlusion_maps/ 顺手删掉
#   4. 校验:   113 个官方场景都在、pair.txt 引用的视角图/相机/深度齐全、图是 768x576
#   5. 删 zip: 只有第 4 步通过才删 (KEEP_ZIP=1 保留)
#   6. 上传:   scripts/upload_blendedmvs_umhpc.sh -> $REMOTE_ROOT/<PID>/, 传完核对文件数
# =============================================================================
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DOWNLOADS=${DOWNLOADS:-/home/william/Downloads}
TARGET=${TARGET:-/home/william/project/dataset/Blended/BlendedMVS_lowres}
REMOTE_HOST=${REMOTE_HOST-qinglong@login01.dicc.um.edu.my}
REMOTE_ROOT=${REMOTE_ROOT:-/scr/user/qinglong/dataset/BlendedMVS_lowres}
ZIPS=${ZIPS:-}
KEEP_ZIP=${KEEP_ZIP:-0}
UPLOAD=${UPLOAD:-1}
POLL=${POLL:-60}
PY=${PY:-$( [[ -x $HOME/miniconda3/envs/uprmvs/bin/python ]] && echo "$HOME/miniconda3/envs/uprmvs/bin/python" || command -v python3)}
EXPECT_HW=${EXPECT_HW:-768x576}
# 期望的场景 = 官方 106/7 列表 (SCENE_LISTS="a.txt b.txt" 可换)
read -r -a LISTS <<< "${SCENE_LISTS:-$REPO/lists/blended/training_list.txt $REPO/lists/blended/validation_list.txt}"
N_EXPECT=$(cat "${LISTS[@]}" | tr -d '\r' | awk 'NF' | sort -u | wc -l)

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 0. 先把 umhpc 连接建好 (输一次密码, 连接保持到脚本结束) -----------------------------
SSH_CTL=$(mktemp -u "${TMPDIR:-/tmp}/blended_ssh.XXXXXX")
export SSH_CTL
close_ssh() { [[ -n "$REMOTE_HOST" ]] && ssh -o ControlPath="$SSH_CTL" -O exit "$REMOTE_HOST" 2>/dev/null || true; }
trap close_ssh EXIT
if [[ "$UPLOAD" == "1" && -n "$REMOTE_HOST" ]]; then
    log "连接 $REMOTE_HOST (输一次密码; 连接会保持到上传结束) ..."
    ssh -fN -o ControlMaster=yes -o ControlPath="$SSH_CTL" -o ControlPersist=yes \
        -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "$REMOTE_HOST"
    ssh -o ControlPath="$SSH_CTL" "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT'"
    log "连接已建立, 之后不用再输密码"
fi

# ---- 校验函数 (第 4 步, 也用来判断能否跳过解压) -----------------------------------
verify_local() {
    "$PY" - "$TARGET" "$EXPECT_HW" "${LISTS[@]}" <<'PY'
import sys
from pathlib import Path
from PIL import Image
root, hw, lists = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
want = list(dict.fromkeys(s.strip() for f in lists for s in open(f).read().split() if s.strip()))
have = {d.name for d in root.iterdir() if d.is_dir()} if root.is_dir() else set()
bad = []
missing = [s for s in want if s not in have]
extra = sorted(have - set(want))
n_views = 0
sizes = {}
for s in want:
    if s in missing:
        continue
    d = root / s
    lines = [l.strip() for l in open(d / "cams/pair.txt") if l.strip()]
    views = set()
    for i in range(int(lines[0])):
        views.add(int(lines[1 + 2 * i]))
        views.update(int(x) for x in lines[2 + 2 * i].split()[1::2])
    n_views += len(views)
    miss = [v for v in views if not ((d / f"blended_images/{v:08d}.jpg").is_file()
            and (d / f"cams/{v:08d}_cam.txt").is_file() and (d / f"rendered_depth_maps/{v:08d}.pfm").is_file())]
    if miss:
        bad.append(f"{s}: {len(miss)} views incomplete")
    with Image.open(d / f"blended_images/{min(views):08d}.jpg") as im:
        sizes[f"{im.width}x{im.height}"] = sizes.get(f"{im.width}x{im.height}", 0) + 1
print(f"[verify] {root}: {len(want) - len(missing)}/{len(want)} scenes, {n_views} views, sizes {sizes}, "
      f"missing {len(missing)}, incomplete {len(bad)}, extra dirs {len(extra)}")
for b in bad[:10] + [f"missing {m}" for m in missing[:10]] + [f"extra {e}" for e in extra[:10]]:
    print("   !!", b)
ok = not missing and not bad and not extra and set(sizes) == {hw}
if set(sizes) - {hw}:
    print(f"   !! image size {sizes} != {hw} (EXPECT_HW=...)")
sys.exit(0 if ok else 1)
PY
}

if verify_local >/dev/null 2>&1; then
    log "$TARGET 已经是完整的 $N_EXPECT 个场景, 跳过等待/解压"
else
    # ---- 1. 等下载完成 -------------------------------------------------------------
    find_zips() {
        if [[ -n "$ZIPS" ]]; then echo "$ZIPS"; return; fi
        find "$DOWNLOADS" -maxdepth 1 -type f \( -name 'BlendedMVS.zip' -o -name 'BlendedMVS.z[0-9][0-9]' \
            -o -iname 'dataset_low_res*.zip' \) | sort
    }
    while :; do
        busy=$(find "$DOWNLOADS" -maxdepth 1 \( -name '*.crdownload' -o -name '*.part' \) | head -3)
        mapfile -t Z < <(find_zips)
        if [[ -z "$busy" && ${#Z[@]} -gt 0 ]]; then break; fi
        log "等待下载: 进行中 [${busy//$'\n'/, }]  已完成的 zip: ${#Z[@]}  (${POLL}s 后再看; 文件名匹配不上就用 ZIPS=... 指定)"
        sleep "$POLL"
    done
    log "压缩包: ${Z[*]}"

    # ---- 2. 合并分卷 + 完整性测试 ------------------------------------------------------
    PARENT=$(dirname "$TARGET")
    mkdir -p "$PARENT"
    need=$(du -cb "${Z[@]}" | tail -1 | cut -f1)
    free=$(df -B1 --output=avail "$PARENT" | tail -1)
    (( free > need * 2 )) || { echo "磁盘不够: 需要约 $((need * 2 / 2**30)) GiB, 可用 $((free / 2**30)) GiB" >&2; exit 1; }
    ARCHIVES=()
    MERGED=""
    for z in "${Z[@]}"; do
        case "$z" in *.z[0-9][0-9]) continue ;; esac          # 分卷的 .z01.. 跟着同名 .zip 走
        base=${z%.zip}
        if compgen -G "$base.z[0-9][0-9]" >/dev/null; then
            MERGED="$PARENT/.merged_$(basename "$base").zip"
            log "合并分卷 $(basename "$base").z* -> $MERGED"
            zip -q -s 0 "$z" --out "$MERGED"
            ARCHIVES+=("$MERGED")
        else
            ARCHIVES+=("$z")
        fi
    done
    for a in "${ARCHIVES[@]}"; do
        log "unzip -t $a (全量校验, 需要几分钟) ..."
        unzip -tq "$a" || { echo "压缩包损坏: $a —— 重新下载" >&2; exit 1; }
    done

    # ---- 3. 解压 + 拍平 -----------------------------------------------------------------
    STAGE="$PARENT/.extract_$(basename "$TARGET")"
    rm -rf "$STAGE"
    mkdir -p "$STAGE" "$TARGET"
    for a in "${ARCHIVES[@]}"; do
        log "解压 $a -> $STAGE"
        unzip -q -o "$a" -d "$STAGE"
    done
    mapfile -t SCENES < <(find "$STAGE" -type d -name blended_images -printf '%h\n' | sort)
    log "找到 ${#SCENES[@]} 个场景目录, 拍平到 $TARGET/<PID>"
    for d in "${SCENES[@]}"; do
        pid=$(basename "$d")
        if [[ -e "$TARGET/$pid" ]]; then
            echo "  $TARGET/$pid 已存在, 用新解压的替换" >&2
            rm -rf "${TARGET:?}/$pid"
        fi
        mv "$d" "$TARGET/$pid"
        [[ -d "$TARGET/$pid/occlusion_maps" ]] && rmdir "$TARGET/$pid/occlusion_maps" 2>/dev/null || true
    done
    left=$(find "$STAGE" -type f | head -5)
    [[ -n "$left" ]] && log "临时目录里剩下的非场景文件 (一并删除): ${left//$'\n'/, }"
    rm -rf "$STAGE"
    [[ -n "$MERGED" ]] && rm -f "$MERGED"

    # ---- 4. 校验 ------------------------------------------------------------------------
    verify_local || { echo "校验没通过, zip 保留, 请检查上面 !! 的行" >&2; exit 1; }

    # ---- 5. 删 zip ------------------------------------------------------------------------
    if [[ "$KEEP_ZIP" == "1" ]]; then
        log "KEEP_ZIP=1, 保留 ${Z[*]}"
    else
        log "校验通过, 删除压缩包: ${Z[*]}"
        rm -f "${Z[@]}"
    fi
fi
verify_local

# ---- 6. 上传 ----------------------------------------------------------------------------
if [[ "$UPLOAD" == "1" ]]; then
    log "上传 $TARGET -> ${REMOTE_HOST:+$REMOTE_HOST:}$REMOTE_ROOT"
    LOCAL_ROOT="$TARGET" REMOTE_HOST="$REMOTE_HOST" REMOTE_ROOT="$REMOTE_ROOT" EXPECT_SCENES="$N_EXPECT" \
        bash "$REPO/scripts/upload_blendedmvs_umhpc.sh"
fi
log "全部完成"
