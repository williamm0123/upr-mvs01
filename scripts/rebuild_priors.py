"""用当前 5 视角管线重建 prior 缓存。

为什么必须重建: 现有缓存是 ``pipeline_version=0`` + ``src_weights`` 形状 (2,),
也就是 1 ref + 2 source 生成的, 而现在训练喂 1 ref + 4 source。VGGT 是多视图
模型, 视角集不同先验就不是同一个东西; 更关键的是 SfM 标尺 —— 3 视角可三角化
的点太少, ``metric_scale_from_sparse`` 的 ``num_pairs < 20`` 兜底被大量触发,
这就是 8.05% 未标尺的来源。实测 5 视角下 num_pairs = 420~2970, 全部有效。

特性:
  * 可续跑 —— 按 ``cache_is_current()`` 的签名跳过已是当前版本的文件, 中断后
    重跑同一条命令即可接着做。
  * --slim —— 不写 ``norm_depth_fill`` (全仓库没有任何代码消费它, 却占 40%
    体积), conf 存 uint8。为上传到集群省带宽。
  * --shard i/N —— 切片, 可以多次分批跑或多卡并行。
  * 进度 + ETA + 失败统计。

用法:
    python scripts/rebuild_priors.py --splits train val
    python scripts/rebuild_priors.py --splits train val --slim
    python scripts/rebuild_priors.py --splits train --shard 0/2      # 前一半
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base.config import build_mvs_config
from data.dtu import DTUMVSDataset
from models.pre_prior import (
    PIPELINE_VERSION,
    PriorPrecomputer,
    cache_signature_from_file,
    save_prior,
    signature_is_current,
)


def _fmt(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    choices=["train", "val", "test"])
    ap.add_argument("--target-w", type=int, default=518)
    ap.add_argument("--target-h", type=int, default=420)
    ap.add_argument("--prior-resize", default="",
                    help="逗号分隔的 split:resize, 例 'train:0.5,val:0.5,test:1.0'。"
                         "留空则用内置默认 —— train/val 0.5 (训练分辨率), "
                         "test 1.0 (全分辨率, 见 test.py ensure_priors 的说明: "
                         "加载时降采样不损失, 升采样造不出细节)")
    ap.add_argument("--slim", action="store_true",
                    help="不写 norm_depth_fill (无人消费), conf 存 uint8 —— 体积约减 55%%")
    ap.add_argument("--shard", default="0/1", help="i/N, 只做第 i 片")
    ap.add_argument("--force", action="store_true", help="忽略签名, 全部重算")
    ap.add_argument("--limit", type=int, default=0, help="只做前 N 个 (调试)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--profile", choices=["local", "umhpc"], default=None)
    ap.add_argument("--num-views", type=int, default=None,
                    help="覆盖 cfg.train.num_views。缓存签名会记下这个值, 训练时"
                         "视角数不一致会被判定过期 —— 所以它必须和最终训练用的"
                         "视角数一致 (local 和 umhpc profile 都是 5)")
    args = ap.parse_args()

    si, sn = (int(x) for x in args.shard.split("/"))
    cfg = build_mvs_config(profile=args.profile)
    nviews = args.num_views or cfg.train.num_views
    twh = (args.target_w, args.target_h)
    dev = torch.device(args.device)

    # 用*原始*列表, 不是 *_clean.txt —— 重建的目的就是把坏 scan 也修好。
    lists = {"train": cfg.paths.train_list_file,
             "val": cfg.paths.val_list_file,
             "test": cfg.paths.test_list_file}

    print(f"[rebuild] === num_views={nviews} (1 ref + {nviews-1} source), "
          f"target_wh={twh}, pipeline_version={PIPELINE_VERSION}, "
          f"slim={args.slim} ===")
    print(f"[rebuild] 训练时若用不同的 num_views, 这批缓存会被判定过期并重建。")
    resize = {"train": 0.5, "val": 0.5, "test": 1.0}
    for kv in filter(None, args.prior_resize.split(",")):
        k, v = kv.split(":"); resize[k.strip()] = float(v)

    jobs = []
    for sp in args.splits:
        ds = DTUMVSDataset(datapath=cfg.paths.dtu_train_root, listfile=str(lists[sp]),
                           nviews=nviews, mode="train" if sp == "train" else "val")
        # build_prior_cache 走 ds.precrop_inputs, 后者按 ds.resize_scale 定图像
        # 大小 —— 所以缓存分辨率由这里决定, 和推理分辨率解耦。
        ds.resize_scale = resize[sp]
        idx = [i for i in range(len(ds)) if i % sn == si]
        jobs.append((sp, ds, idx))
        print(f"[rebuild] {sp}: {len(ds)} 样本, 本片 {len(idx)}, resize_scale={resize[sp]}")

    # 先扫一遍要做多少 —— 装模型很慢, 没活干就不装
    todo = []
    for sp, ds, idx in jobs:
        for i in idx:
            f = Path(ds.prior_cache_path_for(i))
            if args.force or not f.exists():
                todo.append((sp, ds, i)); continue
            sig = cache_signature_from_file(f)   # 只读元数据, 不解压大数组
            if sig is None or not signature_is_current(sig, twh):
                todo.append((sp, ds, i))
            if args.limit and len(todo) >= args.limit:
                break
    if args.limit:
        todo = todo[:args.limit]
    print(f"[rebuild] 需要重建 {len(todo)} 个 (pipeline_version>={PIPELINE_VERSION}, "
          f"num_views={nviews}, target_wh={twh})")
    if not todo:
        print("[rebuild] 全部已是当前版本, 无事可做")
        return

    pc = PriorPrecomputer(dev, image_target_wh=twh)
    t0 = time.time()
    ok = bad = 0
    for n, (sp, ds, i) in enumerate(todo, 1):
        prior = pc.compute(ds.precrop_inputs(i))
        if args.slim:
            # norm_depth_fill: 全仓库没有代码消费它 (dtu.py 产出后无人读), 但占
            # 40% 体积。写一个 1x1x3 占位保持键存在, load_prior 不会 KeyError。
            prior["norm_depth_fill"] = np.zeros((1, 1, 3), np.float32)
            # conf 是 [0,1] 的置信度, uint8 的 1/255 分辨率绰绰有余
            prior["conf_prior"] = (np.clip(prior["conf_prior"], 0, 1) * 255).astype(np.uint8)
        if prior["sfm_valid"] < 0.5:
            bad += 1
        else:
            ok += 1
        save_prior(ds.prior_cache_path_for(i), prior)
        if n % 25 == 0 or n == len(todo):
            el = time.time() - t0
            eta = el / n * (len(todo) - n)
            print(f"[rebuild] {n}/{len(todo)}  {el/n:.2f}s/个  已用 {_fmt(el)}  "
                  f"剩余 {_fmt(eta)}  标尺失败 {bad}", flush=True)
    print(f"\n[rebuild] 完成: {ok} 成功, {bad} 标尺失败 (已写 sfm_valid=0, "
          f"网络会退回 global-only), 用时 {_fmt(time.time()-t0)}")
    if bad:
        print("[rebuild] 标尺失败的样本副本在 log/prior_cache_quarantine/")


if __name__ == "__main__":
    main()
