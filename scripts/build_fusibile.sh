#!/bin/bash
# 编译 fusibile (gipuma 的融合部分) —— test.py --fusion gipuma 依赖它。
#
#     bash scripts/build_fusibile.sh
#
# 它不是本仓库的一部分, 也不在 MVSFormer++ 的参考仓库里 (那边的
# --fusibile_exe_path 默认指向 ./fusibile/fusibile, 需要自己 clone + 编译)。
# 需要 nvcc、cmake、OpenCV。装好后可执行文件在 third_party/fusibile/fusibile。
#
# 为什么非它不可: 内置的 --fusion geo 和 MVSFormer++ 的 dypcd 都是**逐 ref 视角
# 输出存活像素、跨视角不去重** —— 49 个视角下一个 scan 出 2500-4500 万点、ply
# 600MB。fusibile 会把跨视角一致的点聚成一个, 通常降到 1-3M。
#
# 常见坑:
#   * fusibile 的 CMakeLists 里 CUDA arch 往往写死在很老的 sm_30/sm_50, A100 是
#     sm_80 —— 下面用 CUDA_ARCH 覆盖 (默认 80)。
#   * 老代码用了 CUDA 11 之后移除的 texture reference API。若报 texture 相关错误,
#     用较老的 CUDA toolchain, 或改用维护版 fork。
set -e
cd "$(dirname "$0")/.."

CUDA_ARCH=${CUDA_ARCH:-80}          # A100 = 80, V100 = 70, 3090 = 86
REPO=${REPO:-https://github.com/YoYo000/fusibile.git}
DST=third_party/fusibile

command -v nvcc  >/dev/null || {
    echo "找不到 nvcc。登录节点通常没有 CUDA toolkit, 试:" >&2
    echo "    module load cuda/11.8      # 或 module avail cuda 看有哪些" >&2
    echo "    conda install -c nvidia cuda-nvcc=11.8" >&2
    echo "或者申请一个 GPU 节点再编译 (二进制编好之后哪跑都行)。" >&2
    exit 1
}
command -v cmake >/dev/null || { echo "找不到 cmake" >&2; exit 1; }

CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+' | head -1)
echo "=== nvcc: CUDA $CUDA_VER ($(which nvcc)) ==="
echo "=== 目标架构: sm_${CUDA_ARCH} ==="

# fusibile 是 2016 年的代码, 用的是 CUDA 的 texture reference API。那套 API 在
# CUDA 11 里已弃用、**CUDA 12 里彻底移除**, 所以 CUDA >= 12 会在编译期直接失败
# (典型报错: 'texture' is not a template / undefined identifier tex2D)。
# 二进制只要编出来, 在什么 CUDA 运行时上跑都没问题 —— 所以拿一个旧 toolchain
# 编译是可行的, 不用改运行环境。
if [[ -n "$CUDA_VER" && "$CUDA_VER" -ge 12 ]]; then
    echo "" >&2
    echo "!!! CUDA $CUDA_VER: fusibile 用的 texture reference API 在 CUDA 12 已被移除," >&2
    echo "!!! 原版几乎必然编不过。三条出路:" >&2
    echo "!!!   1) 换一个 CUDA 11 的 toolchain 只用来编译:" >&2
    echo "!!!        module load cuda/11.8 && bash scripts/build_fusibile.sh" >&2
    echo "!!!      (编出来的二进制在 CUDA 12 运行时上照常跑)" >&2
    echo "!!!   2) 用修好 texture API 的 fork: REPO=<url> bash scripts/build_fusibile.sh" >&2
    echo "!!!   3) 不用 fusibile, 改用内置的去重融合 (见 --fusion 的说明)" >&2
    echo "" >&2
    echo "仍要继续请设 FORCE=1。" >&2
    [[ "${FORCE:-0}" == "1" ]] || exit 1
fi

mkdir -p third_party
if [[ ! -f "$DST/CMakeLists.txt" ]]; then
    # 只剩 CMakeFiles/ 的残留目录要清掉, 否则 cmake 会拿旧缓存继续报错
    [[ -d "$DST" ]] && { echo "=== 清掉不完整的 $DST ==="; rm -rf "$DST"; }
    git clone --depth 1 "$REPO" "$DST"
fi
[[ -f "$DST/CMakeLists.txt" ]] || { echo "clone 之后仍找不到 $DST/CMakeLists.txt" >&2; exit 1; }

cd "$DST"
# 覆盖写死的 arch。原文件里通常是 -gencode arch=compute_30,code=sm_30 之类。
if grep -q "arch=compute_" CMakeLists.txt 2>/dev/null; then
    sed -i -E "s/arch=compute_[0-9]+,code=sm_[0-9]+/arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}/g" CMakeLists.txt
    echo "=== 已把 CMakeLists.txt 的 CUDA arch 改成 sm_${CUDA_ARCH} ==="
fi

rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

BIN=$(find . -maxdepth 2 -type f -name fusibile -perm -u+x | head -1)
[[ -n "$BIN" ]] || { echo "编译完成但没找到 fusibile 可执行文件" >&2; exit 1; }
cp "$BIN" ../fusibile
cd ../../..
echo
echo "=== 完成: $(pwd)/$DST/fusibile ==="
echo "用法: python test.py ... --fusion gipuma"
