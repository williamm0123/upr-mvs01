"""DTU file readers."""

from __future__ import annotations

import re

import numpy as np


def read_pfm(filename: str) -> np.ndarray:
    """Read a PFM file and return a contiguous numpy array."""
    with open(filename, "rb") as file:
        header = file.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
        if not dim_match:
            raise ValueError("Malformed PFM header.")
        width, height = map(int, dim_match.groups())

        scale = float(file.readline().decode("utf-8").rstrip())
        endian = "<" if scale < 0 else ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        return np.flipud(data).copy()

def write_pfm(filename, data):
    """
    写入 PFM 格式文件（对标 read_pfm）
    支持：单通道灰度图(depth) / 3通道彩色图
    默认使用小端序(little-endian)，兼容 DTU 数据集
    """
    with open(filename, 'wb') as file:
        # 判断是彩色图还是深度图
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        if len(data.shape) == 3 and data.shape[-1] == 3:
            # 彩色图 PF
            file.write(b'PF\n')
            height, width = data.shape[:2]
        else:
            # 深度图 / 灰度图 Pf
            file.write(b'Pf\n')
            height, width = data.shape[:2]

        # 写入宽高
        file.write(f"{width} {height}\n".encode('utf-8'))

        # 小端序（DTU 标准格式）
        scale = 1.0
        file.write(f"-{scale}\n".encode('utf-8'))

        # pfm 文件存储是上下颠倒的，必须 flip
        data = np.flipud(data)

        # 写入二进制数据
        data.tofile(file)


