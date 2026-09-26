#!/bin/bash
# =============================================================================
# 把本地解压好的 BlendedMVS 高分辨率版 (dataset_full_res_*, 113 场景, 2048x1536) 上传到 umhpc,
# 并在上传时把目录拍平成  <REMOTE_ROOT>/<PID>/{blended_images,cams,rendered_depth_maps}。
#
#   bash scripts/upload_blendedmvs_umhpc.sh              # 先检查 + 上传 (断点续传, 可重复跑)
#   DRY_RUN=1 bash scripts/upload_blendedmvs_umhpc.sh    # 只列出要传什么, 不传
#   VERIFY_ONLY=1 bash scripts/upload_blendedmvs_umhpc.sh  # 只做远端文件数核对
#
# 本地的层级不统一: dataset_full_res_30-59/<PID>/<PID>/<PID>/ 是 3 层,
# dataset_full_res_0-29/dataset_full_res_0-29/<PID>/<PID>/<PID>/ 多一层。脚本不猜层数,
# 而是找出每个 blended_images/ 的父目录当作场景目录, 目录名就是 PID。
#
# 做法: 在临时目录里建 <PID> -> 真实场景目录 的符号链接, 再用一条 rsync -L (跟随链接) 传过去,
# 所以只开一个 ssh 连接, 进度是整体的, 中断后原样重跑只补没传完的文件 (--partial)。
# 不传: 每个场景里的 occlusion_maps/ (本地检查全是空目录)。
# *_masked.jpg 默认一起传 (官方数据的一部分, 约占图像的一半、总量的 ~4%); SKIP_MASKED=1 可不传。
#
# 总量约 232 GB, 建议在 tmux / nohup 里跑:
#   nohup bash scripts/upload_blendedmvs_umhpc.sh > upload_blended.log 2>&1 &
# =============================================================================
set -euo pipefail

LOCAL_ROOT=${LOCAL_ROOT:-/media/william/Data2/documents/DATASET/Blended}
REMOTE_HOST=${REMOTE_HOST-qinglong@login01.dicc.um.edu.my}    # 显式设成空 = 本地路径 (测试用)
REMOTE_ROOT=${REMOTE_ROOT:-/scr/user/qinglong/dataset/BlendedMVS}
EXPECT_SCENES=${EXPECT_SCENES:-113}
SKIP_MASKED=${SKIP_MASKED:-0}
DRY_RUN=${DRY_RUN:-0}
VERIFY_ONLY=${VERIFY_ONLY:-0}

# 所有 ssh / rsync 复用同一个连接: 密码登录时只需输一次密码
SSH_CTL=$(mktemp -u "${TMPDIR:-/tmp}/blended_ssh.XXXXXX")
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$SSH_CTL" -o ControlPersist=600)
cleanup() {
    [[ -n "$REMOTE_HOST" ]] && ssh "${SSH_OPTS[@]}" -O exit "$REMOTE_HOST" 2>/dev/null || true
    [[ -n "${STAGE:-}" ]] && rm -rf "$STAGE"
}
trap cleanup EXIT
remote() {   # 在目标机器上跑一条 shell 命令
    if [[ -n "$REMOTE_HOST" ]]; then ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$1"; else bash -c "$1"; fi
}
DEST=${REMOTE_HOST:+$REMOTE_HOST:}$REMOTE_ROOT

# ---- 1. 找场景 -----------------------------------------------------------------
mapfile -t SCENE_DIRS < <(find "$LOCAL_ROOT" -type d -name blended_images -printf '%h\n' | sort)
declare -A SEEN=()
for d in "${SCENE_DIRS[@]}"; do
    pid=$(basename "$d")
    [[ -n "${SEEN[$pid]:-}" ]] && { printf "重复的 PID %s:\n  %s\n  %s\n" "$pid" "${SEEN[$pid]}" "$d" >&2; exit 1; }
    SEEN[$pid]=$d
    for sub in cams rendered_depth_maps; do
        [[ -d "$d/$sub" ]] || { echo "$d 缺 $sub/" >&2; exit 1; }
    done
done
N=${#SCENE_DIRS[@]}
N_PFM=$(find "$LOCAL_ROOT" -path '*/rendered_depth_maps/*.pfm' | wc -l)
N_CAM=$(find "$LOCAL_ROOT" -path '*/cams/*_cam.txt' | wc -l)
N_JPG=$(find "$LOCAL_ROOT" -path '*/blended_images/*.jpg' ! -name '*_masked.jpg' | wc -l)
echo "[local] $LOCAL_ROOT: $N 个场景, $N_JPG 张图, $N_CAM 个相机, $N_PFM 张深度图"
[[ $N -eq $EXPECT_SCENES ]] || { echo "场景数 $N != $EXPECT_SCENES (EXPECT_SCENES=... 可改)" >&2; exit 1; }

verify() {
    echo "[verify] 核对 $DEST ..."
    local out
    out=$(remote "cd '$REMOTE_ROOT' && echo \$(find . -mindepth 1 -maxdepth 1 -type d | wc -l) \
        \$(find . -mindepth 3 -maxdepth 3 -path '*/blended_images/*.jpg' ! -name '*_masked.jpg' | wc -l) \
        \$(find . -mindepth 3 -maxdepth 3 -path '*/cams/*_cam.txt' | wc -l) \
        \$(find . -mindepth 3 -maxdepth 3 -path '*/rendered_depth_maps/*.pfm' | wc -l) \
        \$(find . -name '*.partial' -o -name '.*.??????' | wc -l)")
    read -r r_dirs r_jpg r_cam r_pfm r_tmp <<< "$out"
    echo "[verify] 远端: $r_dirs 个场景目录, $r_jpg 张图, $r_cam 个相机, $r_pfm 张深度图, 残留临时文件 $r_tmp"
    if [[ $r_dirs -eq $N && $r_jpg -eq $N_JPG && $r_cam -eq $N_CAM && $r_pfm -eq $N_PFM && $r_tmp -eq 0 ]]; then
        echo "[verify] OK —— 与本地一致, 结构为 $REMOTE_ROOT/<PID>/{blended_images,cams,rendered_depth_maps}"
    else
        echo "[verify] 不一致 —— 重新运行本脚本即可续传" >&2
        return 1
    fi
}
if [[ "$VERIFY_ONLY" == "1" ]]; then verify; exit $?; fi

# ---- 2. 目标目录必须是干净的 (只允许已有的 PID 场景目录, 方便续传) --------------------
remote "mkdir -p '$REMOTE_ROOT'"
FOREIGN=$(remote "cd '$REMOTE_ROOT' && for e in * .[!.]*; do [ -e \"\$e\" ] || continue; \
    case \"\$e\" in ????????????????????????) [ -d \"\$e\" ] && continue;; esac; echo \"\$e\"; done")
if [[ -n "$FOREIGN" ]]; then
    echo "目标 $REMOTE_ROOT 里有不是场景目录的东西, 为保证 '干净的 BlendedMVS/<PID>' 结构先不传:" >&2
    echo "$FOREIGN" | head -20 | sed 's/^/    /' >&2
    echo "先挪走它们 (例如 BlendedMVS+ 的分卷 BlendedMVS1.z*), 或用 REMOTE_ROOT=... 换一个目录" >&2
    exit 2
fi

# ---- 3. 符号链接暂存 + 一次 rsync ------------------------------------------------
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/blended_stage.XXXXXX")
for d in "${SCENE_DIRS[@]}"; do ln -s "$d" "$STAGE/$(basename "$d")"; done

RSYNC=(rsync -a -L -h --partial --info=progress2,stats1 --exclude 'occlusion_maps/'
       -e "ssh -o ControlMaster=auto -o ControlPath=$SSH_CTL -o ControlPersist=600")
[[ "$SKIP_MASKED" == "1" ]] && RSYNC+=(--exclude '*_masked.jpg')
[[ "$DRY_RUN" == "1" ]] && RSYNC+=(--dry-run --info=name1)
echo "[upload] ${RSYNC[*]} $STAGE/ -> $DEST/"
"${RSYNC[@]}" "$STAGE/" "$DEST/"
[[ "$DRY_RUN" == "1" ]] && { echo "[upload] dry-run 结束, 没有传任何文件"; exit 0; }

# ---- 4. 核对 --------------------------------------------------------------------
verify
