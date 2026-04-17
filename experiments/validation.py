"""Lightweight sample alignment checks from the notebook debug cells."""

from __future__ import annotations

from pathlib import Path

import torch

from data.dtu import expected_camera_path, expected_depth_path, expected_image_path


def tensor_to_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def tensor_to_list(value) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().view(-1).tolist()]
    return [int(x) for x in value]


def build_sample_alignment_report(dataset, sample: dict, sample_index: int) -> dict:
    scan_name, ref_view_from_index, src_views_from_index = dataset.samples[sample_index]
    ref_view = tensor_to_int(sample["ref_view"])
    src_views = tensor_to_list(sample["src_views"])
    view_ids = tensor_to_list(sample["view_ids"])
    light_id = tensor_to_int(sample["light_id"])

    image_paths = [
        expected_image_path(dataset.root_dir, scan_name, dataset.split, dataset.image_dir, light_id, view_id)
        for view_id in view_ids
    ]
    camera_paths = [expected_camera_path(dataset.root_dir, view_id) for view_id in view_ids]
    depth_path = expected_depth_path(dataset.root_dir, scan_name, dataset.split, dataset.depth_dir, ref_view)
    expected_src_views = dataset.view_pairs[ref_view][: dataset.n_views - 1]

    return {
        "sample_index": sample_index,
        "sample_name": sample["sample_name"],
        "scan_name": scan_name,
        "ref_view": ref_view,
        "ref_view_matches_index": ref_view == ref_view_from_index,
        "src_views": src_views,
        "expected_src_views": expected_src_views,
        "src_views_match_pair_file": src_views == expected_src_views,
        "view_ids": view_ids,
        "view_ids_match": view_ids == [ref_view] + src_views,
        "depth_gt_shape": tuple(sample["depth_gt"].shape),
        "imgs_shape": tuple(sample["imgs"].shape),
        "intrinsics_shape": tuple(sample["intrinsics"].shape),
        "extrinsics_shape": tuple(sample["extrinsics"].shape),
        "projection_shape": tuple(sample["projection_matrices"].shape),
        "depth_path": depth_path,
        "depth_path_exists": Path(depth_path).exists(),
        "image_paths": image_paths,
        "image_paths_exist": [Path(path).exists() for path in image_paths],
        "camera_paths": camera_paths,
        "camera_paths_exist": [Path(path).exists() for path in camera_paths],
    }


def print_sample_alignment_report(report: dict) -> None:
    print("=" * 80)
    print("sample_index :", report["sample_index"])
    print("sample_name  :", report["sample_name"])
    print("scan_name    :", report["scan_name"])
    print("ref_view     :", report["ref_view"])
    print("src_views    :", report["src_views"])
    print("view_ids     :", report["view_ids"])
    print("imgs shape   :", report["imgs_shape"])
    print("depth shape  :", report["depth_gt_shape"])
    print("intrinsics   :", report["intrinsics_shape"])
    print("extrinsics   :", report["extrinsics_shape"])
    print("projection   :", report["projection_shape"])
    print("ref matches  :", report["ref_view_matches_index"])
    print("src matches  :", report["src_views_match_pair_file"])
    print("view order ok:", report["view_ids_match"])
    print("depth exists :", report["depth_path_exists"], report["depth_path"])
    print("images exist :", all(report["image_paths_exist"]))
    for path, exists in zip(report["image_paths"], report["image_paths_exist"]):
        print(f" - image {exists}: {path}")
    print("cameras exist:", all(report["camera_paths_exist"]))
    for path, exists in zip(report["camera_paths"], report["camera_paths_exist"]):
        print(f" - camera {exists}: {path}")
