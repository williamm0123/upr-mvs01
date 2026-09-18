#!/usr/bin/env python
"""fusibile 点云融合 —— 把推理缓存下来的逐视角深度融成每个 scan 一个 ply。

在 uprmvs 环境里跑, 分两步:

    # 1) 推理: 只出逐视角深度缓存, 不融合
    python test.py --split test --build-priors skip --fusion none --out log/depth_cache/test
    #    (**不是** --no-fuse —— 那个模式只算深度指标, 一个 npz 都不写。
    #     照常跑 scripts/test_dtu.sh 也行, 它同样会写这份缓存。
    #     一条龙: scripts/test_dtu_fusibile.sh)

    # 2) 融合
    python points_fusibile.py --out log/depth_cache/test --ply-dir log/pred_points_fusibile

为什么单独一个脚本, 而不是 test.py --fuse-only --fusion gipuma:
  * 融合与推理的失败模式完全不相干 —— 推理要模型/先验缓存/显存, 融合要一个外部
    CUDA 二进制和**它自己的运行时**。混在一个进程里, 融合挂了要连 config、
    dataset、先验缓存一起重新走一遍才能重试。
  * 换阈值重融一次不该重跑 1078 个前向。
  * test.py 的 gipuma 分支明确拒绝 --photo-keep-ratio (那条注释说"光度门在
    fusibile 自己的二进制里")。其实不然: 概率过滤是**我们**在导出前做的, 固定
    保留率完全可以在这里实现 —— 见下面的 apply_photo_gate。既有协议 (
    scripts/sbatch_test_dtu.sh) 用的就是 PHOTO_KEEP_RATIO=0.60, 融合后端换成
    fusibile 之后这个自由度不该跟着消失。

fusibile 可执行文件的位置 (按序尝试, 也可以 --fusibile-exe / FUSIBILE_EXE 指定):
    umhpc   /scr/user/qinglong/projects/fusibile/build/fusibile
    本地    /home/william/project/fusibile/build/fusibile
    仓库内  third_party/fusibile/fusibile     (scripts/build_fusibile.sh 的产物)

**跨 conda 环境的坑**: fusibile 编在 fusibile_build 环境里 (OpenCV、libstdc++、
libgomp 都来自那个 env), 模型跑在 uprmvs 里。编译时 CMake 若把 RPATH 写进了 ELF,
它自己就能找到那些库; 否则 uprmvs 的 LD_LIBRARY_PATH 会抢在前面, 典型报错是
  libopencv_core.so.413: cannot open shared object file
  /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found
本脚本在跑之前用 ldd 验一遍, 并把 ELF 里记的 RPATH/RUNPATH 显式提到
LD_LIBRARY_PATH 最前面。还不行就:
    --lib-dir ~/miniconda3/envs/fusibile_build/lib     (或 FUSIBILE_LIB_DIR=...)

流程与 MVSFormer++ misc/gipuma.py 逐字一致 (便于对齐它们的 DTU 数):
  1. 概率过滤: conf 不达标的像素深度置 0 —— fusibile 把 0 当无效
  2. 导出 gipuma 目录树:
        <tmp>/cams/<id>.png.P          P = K_4x4 @ E 的前三行
        <tmp>/images/<id>.png
        <tmp>/2333__<id>/disp.dmb      深度
        <tmp>/2333__<id>/normals.dmb   伪法向 (配 --normal-thresh 360 = 关掉法向检查)
  3. 调 fusibile 二进制
  4. 取 consistencyCheck-*/final3d_model.ply -> <ply-dir>/mvsnet<scan:03d>_l3.ply
     (Fast-DTU-Evaluation 认的命名)

fusibile 有两种**静默**失败, 本脚本都拦下来了 (这也是为什么它比一层薄封装长):
  * CUDA 出错时它只 printf 一句 "Error: ...", 既不停也不返回非零, 照样写出一个
    0 顶点的 ply 然后退出 0。22 个 scan 全空、退出码全是 0、Fast-DTU-Evaluation
    照样给你一组数 —— 典型触发条件是二进制编的 CUDA arch 不含这张卡
    (CMakeLists 里的 CUDA_ARCHITECTURES: A100=80, 4090/L40S=89, H100=90,
    50 系=120)。
  * 进程被 OOM-killer 收走时 ply 的 header 是完整的、数据是截断的, 顶点数照读。
默认这两种情况直接失败; 0 点确实是预期的话用 --allow-empty。

打分仍然是独立的第三步, 本脚本不掺和。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# dmb / P 文件的二进制格式只留一份实现 —— 写错了不会报错, 只会得到一张乱掉的图。
from utils.fusion_gipuma import (  # noqa: E402
    fake_gipuma_normal, write_gipuma_cam, write_gipuma_dmb,
)

# 三处默认位置。第一个是 umhpc 上的, 也是这个脚本的主要目标。
EXE_CANDIDATES = (
    "/scr/user/qinglong/projects/fusibile/build/fusibile",
    "/scr/user/qinglong/projects/fusibile/fusibile",
    "/home/william/project/fusibile/build/fusibile",
    "/home/william/project/fusibile/fusibile",
    str(REPO_ROOT / "third_party/fusibile/fusibile"),
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "fusibile point-cloud fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("流程与")[0])
    # --- 输入 / 输出 ---
    p.add_argument("--out", "--depth-cache", dest="out", default=None,
                   help="推理写的逐视角缓存根目录, 里面应有 depth/<scan>/*.npz "
                        "(直接给 depth/ 那一层也认)。默认 "
                        "cfg.paths.depth_cache_path/<split>。相对路径按项目根解析。")
    p.add_argument("--split", choices=["val", "test"], default="test",
                   help="只用来拼 --out 的默认路径")
    p.add_argument("--ply-dir", default=None,
                   help="融出来的 ply 去处, 默认 cfg.paths.pred_points_path "
                        "(<project>/log/pred_points)。相对路径按项目根解析。")
    p.add_argument("--scans", type=int, nargs="+", default=None, metavar="ID",
                   help="只融这几个 scan (例如 --scans 1 4 9)。ply 按 scan 编号命名, "
                        "所以几个作业可以共用同一个 --ply-dir。")
    p.add_argument("--tmp-dir", default=None,
                   help="导出的 gipuma 目录树放哪, 默认 <out>/fusibile_tmp。"
                        "每个 scan 约 1GB (49 视角 x (disp 4.9MB + normals 14.7MB + png)), "
                        "跑完即删。注意**不要**放进 <out>/depth/ —— 那层是按目录名枚举 "
                        "scan 的, 残留的临时目录会被当成一个 scan。")
    p.add_argument("--keep-tmp", action="store_true", help="保留临时目录 (调试用)")
    p.add_argument("--skip-existing", action="store_true",
                   help="目标 ply 已存在就跳过 (断点续跑)")
    p.add_argument("--workers", type=int, default=4,
                   help="导出阶段的线程数 (npz 解压 + png/dmb 落盘)")

    # --- 光度门 (概率过滤, 在导出前做) ---
    p.add_argument("--photo-thresh", type=float, default=0.3,
                   help="conf <= 该值的像素深度置 0 (MVSFormer++ 用 0.5)")
    p.add_argument("--photo-keep-ratio", type=float, default=0.0,
                   help="固定保留率的光度门: >0 时**忽略** --photo-thresh, 每个参考"
                        "视角精确保留置信度最高的 ceil(r x N_valid) 个像素。"
                        "两个模型的置信度分布不同, 同一个阈值下保留率不同 —— 那时"
                        "比较的就不只是深度质量。既有协议用 0.60。")

    # --- fusibile 本体 ---
    p.add_argument("--fusibile-exe", default=os.environ.get("FUSIBILE_EXE"),
                   help=f"fusibile 可执行文件。默认按序找: {', '.join(EXE_CANDIDATES)}")
    p.add_argument("--lib-dir", action="append", default=None, metavar="DIR",
                   help="加到 LD_LIBRARY_PATH 最前面 (可重复)。跨 conda 环境时用它指向"
                        "编译 fusibile 那个 env 的 lib/。也可用 FUSIBILE_LIB_DIR "
                        "(冒号分隔)。")
    p.add_argument("--clean-ld-path", action="store_true",
                   help="彻底丢掉继承来的 LD_LIBRARY_PATH, 只留 --lib-dir 与 ELF 里的 "
                        "RPATH/RUNPATH。当前环境注入的 libstdc++/libgomp 与 fusibile "
                        "编译环境冲突时用它。")
    p.add_argument("--disp-thresh", type=float, default=0.25,
                   help="视差一致性阈值 (MVSFormer++ 用 0.25)")
    p.add_argument("--num-consistent", type=int, default=3,
                   help="要求的一致视角数 (MVSFormer++ 用 3)")
    p.add_argument("--normal-thresh", type=float, default=360.0,
                   help="法向一致性阈值。伪法向处处相同, 360 = 关掉这一项, 只留视差"
                        "与视角数两个条件 —— 这是 MVSFormer++ 的做法, 改小它没有意义。")
    p.add_argument("--depth-min", type=float, default=0.001)
    p.add_argument("--depth-max", type=float, default=100000.0)
    p.add_argument("--no-color", action="store_true",
                   help="不传 -color_processing (fusibile 的 ply 顶点色本来就是灰度)")
    p.add_argument("--timeout", type=float, default=0.0,
                   help="单个 scan 的 fusibile 超时秒数, 0 = 不限")

    # --- 其它 ---
    p.add_argument("--allow-no-gpu", action="store_true",
                   help="跳过 GPU 预检。fusibile 是 CUDA 程序, 没卡必然失败 —— 这个"
                        "开关只在预检本身不可靠 (例如容器里没有 nvidia-smi) 时用。")
    p.add_argument("--allow-empty", action="store_true",
                   help="允许某个 scan 融出 0 个点。默认这是硬失败 —— 空 ply 会被"
                        "Fast-DTU-Evaluation 照常打分, 悄悄污染一整组数。")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印计划与将要执行的命令行, 不导出也不跑")
    p.add_argument("--verbose", action="store_true",
                   help="把 fusibile 的输出直接打到终端 (默认只写日志文件)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 定位二进制 + 运行时环境
#
# 这一段占的篇幅比融合本身还长, 因为它是这条路唯一真正会咬人的地方: 二进制编在
# 一个 conda 环境里, 却要在另一个环境里跑。
# --------------------------------------------------------------------------- #
def resolve_exe(explicit: str | None) -> Path:
    cands = [explicit] if explicit else list(EXE_CANDIDATES)
    for c in cands:
        if not c:
            continue
        p = Path(c)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
    if explicit:
        raise SystemExit(f"找不到 (或不可执行) 指定的 fusibile: {explicit}")
    raise SystemExit(
        "找不到 fusibile 可执行文件, 试过:\n  " + "\n  ".join(EXE_CANDIDATES) +
        "\n用 --fusibile-exe 指定, 或 bash scripts/build_fusibile.sh 编一个。")


def elf_lib_dirs(exe: Path) -> list[str]:
    """从 ELF 的 DT_RPATH / DT_RUNPATH 里把库目录抠出来。

    为什么要显式提出来: DT_RPATH 优先于 LD_LIBRARY_PATH, 但 **DT_RUNPATH 排在它
    后面**。同一份 CMakeLists 在不同 cmake/binutils 上会写出不同的那一个, 于是
    "本地能跑、集群报 GLIBCXX 版本不够" 这类现象只取决于编译机的工具链版本。把
    它提到 LD_LIBRARY_PATH 最前面, 两种情况就一样了。
    """
    if not shutil.which("readelf"):
        return []
    try:
        r = subprocess.run(["readelf", "-d", str(exe)],
                           capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    dirs: list[str] = []
    for m in re.finditer(r"Library (?:rpath|runpath):\s*\[([^\]]*)\]", r.stdout):
        for d in m.group(1).split(":"):
            d = d.strip()
            if not d:
                continue
            # $ORIGIN 是相对二进制自己的位置, 环境变量里放它没用, 这里就地展开。
            d = d.replace("$ORIGIN", str(exe.parent)).replace("${ORIGIN}", str(exe.parent))
            if d not in dirs:
                dirs.append(d)
    return dirs


def build_env(exe: Path, lib_dirs: list[str], clean: bool) -> tuple[dict, list[str]]:
    env = os.environ.copy()
    extra = list(lib_dirs)
    for d in os.environ.get("FUSIBILE_LIB_DIR", "").split(":"):
        if d and d not in extra:
            extra.append(d)
    for d in elf_lib_dirs(exe):
        if d not in extra:
            extra.append(d)
    inherited = "" if clean else env.get("LD_LIBRARY_PATH", "")
    parts = [d for d in extra if d] + ([inherited] if inherited else [])
    if parts:
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    elif clean:
        env.pop("LD_LIBRARY_PATH", None)
    return env, extra


def missing_libs(exe: Path, env: dict) -> list[str]:
    """ldd 里 "=> not found" 的那些。空列表 = 动态链接这一关过了。"""
    if not shutil.which("ldd"):
        return []
    try:
        r = subprocess.run(["ldd", str(exe)], env=env,
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    return [ln.split("=>")[0].strip()
            for ln in r.stdout.splitlines() if "not found" in ln]


def preflight(exe: Path, env: dict, lib_dirs: list[str], allow_no_gpu: bool) -> None:
    print(f"[fusibile] exe = {exe}")
    if lib_dirs:
        print(f"[fusibile] LD_LIBRARY_PATH 前置: {':'.join(lib_dirs)}")
    miss = missing_libs(exe, env)
    if miss:
        raise SystemExit(
            "fusibile 缺共享库, 现在这个环境跑不起来:\n  " + "\n  ".join(miss) +
            "\n\n它是在**另一个 conda 环境**里编的 (OpenCV/libstdc++ 都来自那边), "
            "而 ELF 里没有留下能用的 RPATH。把编译环境的 lib 目录指出来:\n"
            "  python points_fusibile.py --lib-dir ~/miniconda3/envs/fusibile_build/lib ...\n"
            "  (或 export FUSIBILE_LIB_DIR=...; 若报的是 GLIBCXX 版本不够, "
            "再加 --clean-ld-path)")

    # fusibile 是 CUDA 程序。没卡它会在跑到一半时报 CUDA error, 不会静默退回 CPU,
    # 但那时一个 scan 的 1GB 导出已经写完了 —— 不如在这里就停。
    smi = shutil.which("nvidia-smi")
    ok = False
    if smi:
        try:
            r = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30)
            ok = r.returncode == 0 and bool(r.stdout.strip())
            if ok:
                names = ", ".join(x.strip() for x in r.stdout.splitlines() if x.strip())
                print(f"[fusibile] GPU: {names[:160]}")
        except Exception:
            ok = False
    if not ok and not allow_no_gpu:
        raise SystemExit(
            "看不到 GPU (nvidia-smi 不存在或报不出卡)。fusibile 是 CUDA 程序, "
            "登录节点上跑不了 —— 用 sbatch 申请 --gres=gpu:1, 或 --allow-no-gpu 强来。")


def explain_failure(log_tail: str) -> str:
    """把 fusibile 常见的死法翻译成能照做的下一步。"""
    hints = []
    low = log_tail.lower()
    if "no kernel image is available" in low or "unsupported toolchain" in low \
            or "invalid device function" in low:
        hints.append(
            "  * CUDA arch 对不上这张卡 ('no kernel image' / 'PTX ... unsupported "
            "toolchain' / 'invalid device function')。\n"
            "    看 fusibile/CMakeLists.txt 的 CUDA_ARCHITECTURES —— A100=80, "
            "L40S/4090=89, H100=90, 5060Ti/5090=120。\n"
            "    改完在**有 nvcc 的节点**上重编 (二进制编出来之后哪跑都行)。")
    if "out of memory" in low or "cudaerrormemoryallocation" in low:
        hints.append(
            "  * 显存不够: fusibile 一次把整个 scan 的所有视角搬上卡。降 --resize-scale "
            "重跑推理, 或换大卡。")
    if "cannot open shared object" in low or "glibcxx" in low or "glibc_" in low:
        hints.append(
            "  * 动态库对不上: --lib-dir <编译 fusibile 那个 env>/lib, 必要时再加 "
            "--clean-ld-path。")
    if "image seems to be invalid" in low:
        hints.append("  * 'Image seems to be invalid': images/ 里的 png 没写出来或写坏了。")
    if "numimages is 0" in low:
        hints.append(
            "  * 'numImages is 0': fusibile 没认出 2333__<id> 子目录。目录名必须至少两个"
            "下划线且以 '2' 开头, 而且 images/<id>.png 要存在。")
    return "\n" + "\n".join(hints) if hints else ""


# --------------------------------------------------------------------------- #
# 光度门 (概率过滤)
# --------------------------------------------------------------------------- #
def apply_photo_gate(depth: np.ndarray, conf: np.ndarray,
                     thresh: float, keep_ratio: float) -> tuple[np.ndarray, dict]:
    """把不达标的像素深度置 0, 返回 (depth, 统计)。

    非有限值一并清掉 —— nan/inf 写进 dmb 不会报错, 只会让 fusibile 产出垃圾点。
    """
    depth = np.asarray(depth, np.float32)
    conf = np.asarray(conf, np.float32)          # 老缓存里是 float16, 见下面的并列告警
    valid = np.isfinite(depth) & (depth > 0)
    n_valid = int(valid.sum())
    ties = 0
    if keep_ratio > 0.0:
        gate = np.zeros_like(valid)
        if n_valid:
            cv_ = conf[valid]
            k = max(1, min(n_valid, int(math.ceil(keep_ratio * n_valid))))
            # stable: conf 大面积并列时 (product 口径下 1.0 的像素很多) 顺序仍由
            # 像素索引决定 —— 同一份输入必须给出同一片点云。
            order = np.argsort(-cv_, kind="stable")
            sel = np.zeros(n_valid, bool)
            sel[order[:k]] = True
            gate[valid] = sel
            # 名额里有多少是"并列后按索引发的"。这个比例一高, 门就不是按置信度在
            # 选, 而是按光栅顺序 —— 2026-08-08 那组 DTU 数就是这么废掉的。
            thr = cv_[order[k - 1]]
            ties = k - int((cv_ > thr).sum())
    else:
        gate = conf > thresh
    keep = valid & gate
    out = np.where(keep, depth, np.float32(0.0)).astype(np.float32)
    return out, {
        "valid": n_valid, "kept": int(keep.sum()), "ties": ties,
        "conf_min": float(conf.min()) if conf.size else 0.0,
        "conf_max": float(conf.max()) if conf.size else 0.0,
    }


# --------------------------------------------------------------------------- #
# 导出一个 scan
# --------------------------------------------------------------------------- #
def export_scan(scan_dir: Path, tmp: Path, args) -> dict:
    cams, imgs = tmp / "cams", tmp / "images"
    for d in (tmp, cams, imgs):
        d.mkdir(parents=True, exist_ok=True)

    files = sorted(scan_dir.glob("*.npz"))
    shapes: set[tuple[int, int]] = set()

    def one(f: Path) -> dict:
        try:
            z = np.load(f)
            depth_raw, conf_raw, image = z["depth"], z["conf"], np.asarray(z["image"])
            K, E = z["K"], z["E"]
        except Exception as e:
            # 被打断的推理会在缓存里留下截断的 npz (几 KB 而不是几 MB)。原始的
            # zlib/EOF 报错说不出是哪个文件, 而且从线程池里冒出来只剩一屏栈。
            raise SystemExit(
                f"{f} 读不出来: {type(e).__name__}: {e}\n"
                f"  ({f.stat().st_size:,} 字节 —— 正常一个视角是几 MB。多半是上一次推理"
                f"被打断留下的半截文件。)\n"
                f"  删掉它再补跑那个视角, 或者整个 scan 重跑一遍推理。") from None
        depth, st = apply_photo_gate(depth_raw, conf_raw,
                                     args.photo_thresh, args.photo_keep_ratio)
        if image.shape[:2] != depth.shape:
            raise SystemExit(f"{f}: image {image.shape[:2]} 与 depth {depth.shape} "
                             f"尺寸不一致, fusibile 会按 image 定 rows/cols。")
        name = f"{int(f.stem):08d}"
        # fusibile 用 OpenCV 读图, 解出来是 BGR; 缓存里存的是 RGB, 所以写盘前翻一次。
        # 顶点色最后是灰度, 但灰度权重 (0.299R+0.587G+0.114B) 对通道顺序敏感, 而
        # 光度处理用的正是这张灰度图。
        if not cv2.imwrite(str(imgs / f"{name}.png"), image[:, :, ::-1]):
            raise SystemExit(f"写不出 {imgs / (name + '.png')} —— 先看磁盘配额。")
        write_gipuma_cam(cams / f"{name}.png.P", K, E)
        # 目录名必须以 '2' 开头且含 >=2 个下划线, 这是 fusibile 认子目录的条件。
        sub = tmp / f"2333__{name}"
        sub.mkdir(exist_ok=True)
        write_gipuma_dmb(sub / "disp.dmb", depth)
        write_gipuma_dmb(sub / "normals.dmb", fake_gipuma_normal(depth))
        st["hw"] = depth.shape
        return st

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        stats = list(ex.map(one, files))

    for s in stats:
        shapes.add(tuple(s.pop("hw")))
    if len(shapes) > 1:
        raise SystemExit(f"{scan_dir.name}: 同一个 scan 里出现了多种分辨率 {shapes} —— "
                         f"fusibile 用第 0 张图定 rows/cols, 混着会算错。")
    agg = {k: sum(s[k] for s in stats) for k in ("valid", "kept", "ties")}
    agg["views"] = len(stats)
    agg["conf_min"] = min((s["conf_min"] for s in stats), default=0.0)
    agg["conf_max"] = max((s["conf_max"] for s in stats), default=0.0)
    agg["hw"] = list(shapes)[0] if shapes else (0, 0)
    return agg


def gate_report(scan: str, agg: dict, args) -> None:
    valid, kept = agg["valid"], agg["kept"]
    frac = kept / valid if valid else 0.0
    if args.photo_keep_ratio > 0.0:
        tie = agg["ties"] / kept if kept else 0.0
        tag = "OK" if tie < 0.05 else ("注意" if tie < 0.30 else "**门是死的**")
        print(f"[fusibile] {scan}: 光度门 keep_ratio={args.photo_keep_ratio} -> "
              f"保留 {100 * frac:.1f}%, 其中 {100 * tie:.1f}% 的名额落在并列组里 ({tag})")
        if tie >= 0.30:
            print(f"[fusibile] WARNING: {scan} 的保留名额有 {100 * tie:.1f}% 是按像素索引"
                  f"发的, 不是按置信度。这批点云只反映光栅顺序, 先查 conf 分布再用。")
    else:
        print(f"[fusibile] {scan}: 光度门 thresh={args.photo_thresh} -> 保留 {100 * frac:.1f}%")
        if frac > 0.999 or frac < 0.001:
            print(f"[fusibile] WARNING: {scan} 的保留率 {100 * frac:.2f}% —— 这个门实际"
                  f"没起作用 (或全筛掉了), 融合等于只跑 fusibile 的几何一致性。"
                  f"conf 范围 [{agg['conf_min']:.4f}, {agg['conf_max']:.4f}]。"
                  f"这正是 2026-08-08 那组 DTU 数作废的原因。")
    if agg["conf_min"] == agg["conf_max"]:
        print(f"[fusibile] WARNING: {scan} 的 conf 是常数 {agg['conf_min']:.4f} —— "
              f"这份深度缓存的置信度是废的, 换哪种门都一样。")


# --------------------------------------------------------------------------- #
# 调用 fusibile
# --------------------------------------------------------------------------- #
def fusibile_cmd(exe: Path, tmp: Path, args) -> list[str]:
    # 路径末尾的 '/' 不能省: fusibile 是字符串拼接 (results_folder + subfolder)。
    cmd = [str(exe),
           "-input_folder", f"{tmp}/",
           "-p_folder", f"{tmp / 'cams'}/",
           "-images_folder", f"{tmp / 'images'}/",
           f"--depth_min={args.depth_min}",
           f"--depth_max={args.depth_max}",
           f"--normal_thresh={args.normal_thresh}",
           f"--disp_thresh={args.disp_thresh}",
           f"--num_consistent={args.num_consistent}"]
    if not args.no_color:
        cmd.append("-color_processing")
    return cmd


def run_fusibile(exe: Path, tmp: Path, args, env: dict, log_path: Path) -> Path:
    cmd = fusibile_cmd(exe, tmp, args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[fusibile] $ " + " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, env=env, cwd=str(tmp), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace")
        # 看门狗线程, 不是 proc.wait(timeout=)。我们阻塞在下面那个读 stdout 的循环
        # 上, 走不到 wait —— 而 fusibile 真挂住时恰好是"不再输出", 那种情况下
        # wait 的 timeout 一秒都用不上。
        killed = {"by_timeout": False}
        timer = None
        if args.timeout > 0:
            import threading

            def _kill() -> None:
                killed["by_timeout"] = True
                proc.kill()

            timer = threading.Timer(args.timeout, _kill)
            timer.daemon = True
            timer.start()
        tail: list[str] = []
        # fusibile 的 CUDA 错误只是 printf("Error: %s\n", cudaGetErrorString(err)),
        # **既不返回非零也不停下** —— 它照样把一个 0 个顶点的 ply 写出来然后退出 0。
        # 22 个 scan 全空、退出码全是 0、Fast-DTU-Evaluation 照样给你一组数, 这是
        # 这条路上最危险的一种失败。所以要在这里把它捞出来。
        # (日志开头那句 "Command-line parameter error: unknown option -input_folder"
        #  是**正常的**: fusibile 分两轮解析参数, 第一轮不认目录类选项, 而且它的
        #  return -1 被注释掉了。不要去追它。)
        cuda_errs: list[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                tail.append(line)
                if len(tail) > 60:
                    tail.pop(0)
                if line.startswith("Error: "):
                    cuda_errs.append(line.strip())
                if args.verbose:
                    print("  | " + line.rstrip(), flush=True)
            rc = proc.wait()
        finally:
            if timer is not None:
                timer.cancel()
        if killed["by_timeout"]:
            raise SystemExit(f"fusibile 超过 {args.timeout}s 未结束, 已杀掉。日志: {log_path}")
    if rc != 0:
        blob = "".join(tail)
        raise SystemExit(f"fusibile 退出码 {rc}。日志: {log_path}\n--- 末尾 ---\n{blob}"
                         + explain_failure(blob))

    if cuda_errs:
        blob = "\n".join(dict.fromkeys(cuda_errs))
        raise SystemExit(
            f"fusibile 报了 CUDA 错误 (但**退出码 0**, 而且已经写出一个空 ply):\n"
            f"  {blob}\n日志: {log_path}" + explain_failure(blob))

    outs = sorted(tmp.glob("consistencyCheck-*/final3d_model.ply"), key=os.path.getmtime)
    if not outs:
        blob = "".join(tail)
        raise SystemExit(f"fusibile 退出码 0 但没产出 final3d_model.ply。日志: {log_path}\n"
                         f"--- 末尾 ---\n{blob}" + explain_failure(blob))
    return outs[-1]


# --------------------------------------------------------------------------- #
# ply 落盘 + 校验
# --------------------------------------------------------------------------- #
_PLY_TYPE_BYTES = {
    b"char": 1, b"uchar": 1, b"int8": 1, b"uint8": 1,
    b"short": 2, b"ushort": 2, b"int16": 2, b"uint16": 2,
    b"int": 4, b"uint": 4, b"int32": 4, b"uint32": 4, b"float": 4, b"float32": 4,
    b"double": 8, b"float64": 8,
}


def ply_header(path: Path) -> tuple[int, int, int]:
    """返回 (顶点数, header 字节数, 每个顶点字节数); 读不出来给 (-1, -1, -1)。

    stride 是按 header 里的 property 列表算的, 不是写死的 27 —— 写死的话, 哪天
    fusibile 换了输出字段, 下面那个完整性校验就会开始误报。
    """
    n, nbytes, stride = -1, 0, 0
    with path.open("rb") as f:
        for _ in range(64):
            line = f.readline()
            if not line:
                return -1, -1, -1
            nbytes += len(line)
            parts = line.split()
            if len(parts) >= 3 and parts[0] == b"element" and parts[1] == b"vertex":
                n = int(parts[2])
            elif len(parts) >= 3 and parts[0] == b"property":
                stride += _PLY_TYPE_BYTES.get(parts[1], 0)
            if line.strip() == b"end_header":
                return n, nbytes, stride
    return -1, -1, -1


def copy_verified(src: Path, dst: Path) -> None:
    """复制 + fsync + 比字节数。

    网络/配额文件系统上, 缓冲写在配额耗尽时不会立刻报错, 于是留下一个只有文件头的
    ply 而调用方照常打印点数 —— 2026-08-22 那次 dedup 融合就是这么过去的。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    want = src.stat().st_size
    with src.open("rb") as fi, dst.open("wb") as fo:
        shutil.copyfileobj(fi, fo, length=8 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    got = dst.stat().st_size
    if got != want:
        raise SystemExit(f"{dst} 写入不完整: {got} 字节, 应为 {want}。多半是磁盘配额满了 —— "
                         f"先 df / quota 再重跑。")


def scan_id_of(name: str) -> int | None:
    m = re.match(r"scan[_-]?(\d+)", name, re.IGNORECASE) or re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def ply_name(scan: str) -> str:
    sid = scan_id_of(scan)
    # Fast-DTU-Evaluation 认的命名。认不出编号就退回目录名, 不静默拼一个错编号。
    return f"mvsnet{sid:03d}_l3.ply" if sid is not None else f"{scan}.ply"


# --------------------------------------------------------------------------- #
# manifest —— 一个 ply 目录如果说不出它是哪份深度缓存、哪套融合参数产出的,
# 它就不能当基线 (log/pred_points_R_geo_dedup 的教训)。
# --------------------------------------------------------------------------- #
def git_info(root: Path) -> dict:
    out: dict = {}
    for key, cmd in (("sha", ["git", "rev-parse", "HEAD"]),
                     ("dirty", ["git", "status", "--porcelain"])):
        try:
            r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=10)
            out[key] = (r.stdout.strip() if key == "sha" else bool(r.stdout.strip())) \
                if r.returncode == 0 else None
        except Exception:
            out[key] = None
    return out


def write_manifest(ply_dir: Path, exe: Path, cache_root: Path, args,
                   scans: dict, lib_dirs: list[str]) -> None:
    path = ply_dir / "fusibile_manifest.json"
    params = {
        "photo_thresh": args.photo_thresh,
        "photo_keep_ratio": args.photo_keep_ratio,
        "disp_thresh": args.disp_thresh,
        "num_consistent": args.num_consistent,
        "normal_thresh": args.normal_thresh,
        "depth_min": args.depth_min, "depth_max": args.depth_max,
        "color_processing": not args.no_color,
    }
    src_man = cache_root / "run_manifest.json"
    man = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "command": " ".join([sys.executable.split("/")[-1], *sys.argv]),
        "code_git": git_info(REPO_ROOT),
        "fusibile": {"exe": str(exe), "mtime": exe.stat().st_mtime,
                     "bytes": exe.stat().st_size, "lib_dirs": lib_dirs},
        "depth_cache": {
            "path": str(cache_root),
            # 深度那一段的谱系 (checkpoint / prior 缓存 / 分辨率) 全在这里面, 原样
            # 抄进来, 否则这个 ply 目录只知道自己是怎么融的, 不知道深度是谁出的。
            "run_manifest": json.loads(src_man.read_text(encoding="utf-8"))
            if src_man.exists() else None,
        },
        "params": params,
        "scans": scans,
    }
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        if old.get("params") and old["params"] != params:
            print(f"[fusibile] WARNING: {path} 里记的融合参数与本次不同 —— 这个目录会"
                  f"同时装着两套协议产出的 ply, 之后没法说清哪个是哪个。旧参数已存进"
                  f" 'superseded'。\n  旧: {old['params']}\n  新: {params}")
            man["superseded"] = old
        else:
            merged = dict(old.get("scans") or {})
            merged.update(scans)
            man["scans"] = merged
    path.write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    print(f"[fusibile] wrote {path}")


# --------------------------------------------------------------------------- #
def resolve_paths(args) -> tuple[Path, Path]:
    from base.config import ProjectPaths
    paths = ProjectPaths()
    root = Path(paths.project_path)

    out = Path(args.out) if args.out else Path(paths.depth_cache_path) / args.split
    if not out.is_absolute():
        out = root / out
    # 给 <out>/depth 也认: 那是最容易顺手粘过来的一层。
    if out.name == "depth" and out.is_dir():
        out = out.parent
    if not (out / "depth").is_dir():
        raise SystemExit(f"{out} 下面没有 depth/ —— 那是 test.py 写逐视角 npz 的地方。\n"
                         f"先跑推理: python test.py --split {args.split} "
                         f"--build-priors skip --fusion none --out <这个目录>")

    ply = Path(args.ply_dir) if args.ply_dir else Path(paths.pred_points_path)
    if not ply.is_absolute():
        ply = root / ply
    return out, ply


def collect_scans(cache_root: Path, want: list[int] | None) -> list[Path]:
    dirs = []
    for d in sorted((cache_root / "depth").iterdir()):
        # '_' 开头的是临时/残留目录 (老版本把 gipuma 中间目录写在了这一层),
        # 没有 npz 的同理 —— 都不是 scan。
        if not d.is_dir() or d.name.startswith("_") or not any(d.glob("*.npz")):
            continue
        if want is not None and scan_id_of(d.name) not in want:
            continue
        dirs.append(d)
    if not dirs:
        raise SystemExit(f"{cache_root / 'depth'} 下没有可用的 scan 目录"
                         + (f" (--scans {want})" if want else ""))
    return dirs


def main() -> None:
    args = parse_args()
    if args.photo_keep_ratio > 0.0 and args.photo_thresh != 0.3:
        print(f"[fusibile] 注意: --photo-keep-ratio {args.photo_keep_ratio} 生效, "
              f"**忽略** --photo-thresh {args.photo_thresh}")
    cache_root, ply_dir = resolve_paths(args)
    scan_dirs = collect_scans(cache_root, args.scans)
    exe = resolve_exe(args.fusibile_exe)
    env, lib_dirs = build_env(exe, args.lib_dir or [], args.clean_ld_path)
    tmp_root = Path(args.tmp_dir) if args.tmp_dir else cache_root / "fusibile_tmp"
    if not tmp_root.is_absolute():
        tmp_root = REPO_ROOT / tmp_root

    print("=" * 70)
    print(f" fusibile 融合  {len(scan_dirs)} scans")
    print(f" depth cache = {cache_root}")
    print(f" ply         = {ply_dir}")
    print(f" tmp         = {tmp_root}   (跑完即删)" if not args.keep_tmp
          else f" tmp         = {tmp_root}   (--keep-tmp: 保留)")
    print(f" gate        = " + (f"keep_ratio {args.photo_keep_ratio}"
                                if args.photo_keep_ratio > 0 else f"thresh {args.photo_thresh}"))
    print(f" fusibile    = disp {args.disp_thresh}  num_consistent {args.num_consistent}"
          f"  normal_thresh {args.normal_thresh}")
    print("=" * 70)

    if args.dry_run:
        for sd in scan_dirs:
            print(f"  {sd.name}: {len(list(sd.glob('*.npz')))} views -> "
                  f"{ply_dir / ply_name(sd.name)}")
        print("\n[dry-run] 每个 scan 会执行:")
        print("  " + " ".join(fusibile_cmd(exe, tmp_root / "<scan>", args)))
        return

    preflight(exe, env, lib_dirs, args.allow_no_gpu)
    ply_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for sd in scan_dirs:
        dst = ply_dir / ply_name(sd.name)
        if args.skip_existing and dst.exists():
            print(f"[fusibile] {sd.name}: 已存在 {dst}, 跳过 (--skip-existing)")
            continue
        tmp = tmp_root / sd.name
        if tmp.exists():
            shutil.rmtree(tmp)
        t0 = time.time()
        try:
            agg = export_scan(sd, tmp, args)
            print(f"[fusibile] {sd.name}: 导出 {agg['views']} 视角 "
                  f"{agg['hw'][0]}x{agg['hw'][1]}  {time.time() - t0:.1f}s")
            gate_report(sd.name, agg, args)
            if agg["kept"] == 0:
                raise SystemExit(f"{sd.name}: 光度门把所有像素都筛掉了, 没东西可融。")
            t1 = time.time()
            src = run_fusibile(exe, tmp, args, env,
                               ply_dir / "fusibile_logs" / f"{sd.name}.log")
            n, hdr, stride = ply_header(src)
            # fusibile 被 OOM-killer 收走时也会留下一个 header 完整、数据截断的 ply。
            # 顶点数是 header 里写的, 不是数出来的 —— 不比一下字节数就看不出来。
            if n > 0 and stride > 0:
                want = hdr + n * stride
                got = src.stat().st_size
                if got != want:
                    raise SystemExit(
                        f"fusibile 写的 ply 是截断的: {src} 有 {got:,} 字节, header 说"
                        f"应该是 {want:,} ({n:,} 顶点 x {stride} 字节 + {hdr} 头)。"
                        f"多半是进程被杀 (OOM) 或磁盘满了。")
            copy_verified(src, dst)
            secs = time.time() - t0
            mb = dst.stat().st_size / 2 ** 20
            print(f"[fusibile] {sd.name}: {n:,} points -> {dst}  "
                  f"({mb:.1f} MB, 融合 {time.time() - t1:.1f}s, 合计 {secs:.1f}s)")
            if n <= 0:
                msg = (f"{sd.name} 融出 {n} 个点。一个空 ply 会被 Fast-DTU-Evaluation "
                       f"照常打分, 所以这里不往下走。\n"
                       f"  常见原因: --num-consistent {args.num_consistent} 太严; "
                       f"光度门留下的像素跨视角对不上; 相机 P 与深度不同尺度。\n"
                       f"  日志: {ply_dir / 'fusibile_logs' / (sd.name + '.log')}\n"
                       f"  确实预期为空就加 --allow-empty。")
                if not args.allow_empty:
                    raise SystemExit(msg)
                print(f"[fusibile] WARNING: {msg}")
            results[sd.name] = {
                "views": agg["views"], "hw": list(agg["hw"]), "points": n,
                "kept_ratio": round(agg["kept"] / agg["valid"], 6) if agg["valid"] else 0.0,
                "tie_frac": round(agg["ties"] / agg["kept"], 6) if agg["kept"] else 0.0,
                "conf_range": [agg["conf_min"], agg["conf_max"]],
                "ply": str(dst), "bytes": dst.stat().st_size,
                "seconds": round(secs, 1),
            }
        finally:
            if not args.keep_tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    if not args.keep_tmp:
        # 空壳目录清掉, 免得下次被当成 scan (它在 <out>/ 下, 不在 depth/ 里, 但还是
        # 别留)。非空说明有 --keep-tmp 的残留, 那就留着。
        try:
            tmp_root.rmdir()
        except OSError:
            pass

    if results:
        write_manifest(ply_dir, exe, cache_root, args, results, lib_dirs)
        total = sum(r["points"] for r in results.values() if r["points"] > 0)
        print(f"\n[fusibile] 完成 {len(results)} 个 scan, 共 {total:,} 点 -> {ply_dir}")
    print("\n打分是独立的第三步:\n"
          f"  cd <Fast-DTU-Evaluation> && python eval_dtu.py --method mvsnet --save \\\n"
          f"      --pred_dir {ply_dir} --gt_dir <DTU GT root>")


if __name__ == "__main__":
    main()
