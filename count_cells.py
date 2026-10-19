# -*- coding: utf-8 -*-
"""
批量识别并统计显微镜照片中的细胞数量。

目录中同时包含两类照片：
  1. 荧光照片（绿色荧光细胞，暗背景）      -> 按 G-R 绿色信号阈值分割 + 距离变换分水岭计数
  2. 光学/相差照片（亮背景，细胞有亮晕）   -> 多尺度“亮环”滤波 + 局部极大值计数（每个细胞一个峰）

脚本会自动判断每张照片属于哪一类，并将 file name / modality / cell_count 写入 CSV。
默认处理脚本所在目录下的 jpg/jpeg/png/tif/tiff 文件（跳过以 "_" 开头的临时文件）。

用法：
    python count_cells.py
    python count_cells.py --input-dir <图片目录> --output <输出.csv> --overlays

注意：
  * 荧光照片的计数较可靠；相差照片在细胞密集、成团或严重贴壁铺展时，
    自动计数只能给出近似值（可能漏掉边界不清的细胞，或把大细胞拆成两个）。
  * 所有照片按统一比例缩放到宽 2000 像素后处理，因此阈值不受原始 5440x3648 分辨率影响。
"""

import argparse
import csv
import os
import re
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# ---------------------------------------------------------------- 基本设置
PROC_W, PROC_H = 2000, 1341          # 统一处理尺寸
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# 相差照片用的多尺度环状卷积核（亮环 - 暗中心）
RING_RADII = [9, 11, 13, 15, 17, 19, 22, 25, 28, 31]


def make_ring_kernels():
    kernels = []
    for r in RING_RADII:
        w = 4                                   # 环宽
        R = r + w + 1
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        d = np.sqrt(xx * xx + yy * yy)
        ring = ((d >= r - w / 2) & (d <= r + w / 2)).astype(np.float32)
        ring /= ring.sum()
        inner = (d <= r - w).astype(np.float32)
        inner /= inner.sum()
        kernels.append(ring - inner)
    return kernels


RING_KERNELS = make_ring_kernels()
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# ---------------------------------------------------------------- 读图 / 缩放
def read_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("无法读取图片: %s" % path)
    return img


def resize_for_processing(img):
    return cv2.resize(img, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 类型判断
def detect_modality(img):
    """荧光图: 背景很暗且绿色明显强于红色; 其余按相差/明场处理。"""
    small = cv2.resize(img, (400, 268), interpolation=cv2.INTER_AREA)
    b, g, r = [float(np.median(small[:, :, i])) for i in range(3)]
    if b < 40 and g > r + 5:
        return "fluorescence"
    return "brightfield"


# ---------------------------------------------------------------- 荧光计数
def count_fluorescence(img, min_area=250, peak_dist=14):
    """
    G-R 去掉黄色标尺，减去大尺度背景后阈值分割；
    再用距离变换 + 分水岭把相连细胞分开，过滤过小或过暗的区域。
    返回 (细胞数, 用于画图的标签图, 前景掩膜)。
    """
    im = resize_for_processing(img)
    B = im[:, :, 0].astype(np.float32)
    G = im[:, :, 1].astype(np.float32)
    R = im[:, :, 2].astype(np.float32)
    sig = np.clip(G - R, 0, None)                       # 绿色荧光信号

    hp = sig - cv2.GaussianBlur(sig, (0, 0), 60)        # 去不均匀背景
    hf = sig - cv2.GaussianBlur(sig, (0, 0), 2)         # 估噪声
    sigma = max(1.4826 * np.median(np.abs(hf - np.median(hf))), 0.3)
    T = max(10.0 * sigma, 5.0)                          # 区域阈值
    T2 = max(15.0 * sigma, 8.0)                         # 单细胞亮度阈值

    fg = hp > T
    fg = ndi.binary_closing(fg, iterations=2)
    fg = ndi.binary_fill_holes(fg)
    # 去掉面积过小的噪点（相当于 remove_small_objects）
    comp, _ = ndi.label(fg)
    sizes = np.bincount(comp.ravel())
    big = sizes >= min_area
    big[0] = False
    fg = big[comp]

    dist = ndi.distance_transform_edt(fg)
    coords = peak_local_max(dist, min_distance=peak_dist,
                            labels=fg, exclude_border=False)
    markers = np.zeros(fg.shape, np.int32)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i
    labels = watershed(-dist, markers, mask=fg)

    keep = []
    for i in range(1, labels.max() + 1):
        m = labels == i
        if int(m.sum()) < min_area:
            continue
        if float(hp[m].max()) < T2:
            continue
        keep.append(i)

    return len(keep), labels, keep


# ---------------------------------------------------------------- 相差/明场计数
def count_brightfield(img, min_dist=22):
    """
    多尺度亮环滤波：细胞在相差图里表现为“亮边缘 + 较暗中心”，
    环状核对此模式响应最强；取所有尺度中的最大响应，再找局部极大值。
    每个极大值视为一个细胞。阈值按本图响应的 95 分位自适应，
    并设置绝对下限以抑制空白图的噪声。
    返回 (细胞数, 峰值点坐标数组)。
    """
    im = resize_for_processing(img)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = CLAHE.apply(gray)
    gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2)

    resp = None
    for k in RING_KERNELS:
        r = cv2.filter2D(gray, -1, k)
        resp = r if resp is None else np.maximum(resp, r)
    resp = cv2.GaussianBlur(resp, (0, 0), 4)

    h, w = resp.shape
    valid = np.ones_like(resp, dtype=bool)
    # 右下角是黄色标尺和 "10X" 字样，排除以避免误检
    valid[int(0.88 * h):, int(0.79 * w):] = False

    thr = max(3.0, float(np.percentile(resp[valid], 95)))
    pts = peak_local_max(resp, min_distance=min_dist,
                         threshold_abs=thr, exclude_border=False)
    pts = pts[valid[pts[:, 0], pts[:, 1]]]

    return len(pts), pts


# ---------------------------------------------------------------- 画标注图
def save_overlay(img, modality, result, path):
    im = resize_for_processing(img)
    if modality == "fluorescence":
        _, labels, keep = result
        for i in keep:
            m = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, contours, -1, (0, 0, 255), 2)
    else:
        _, pts = result
        for y, x in pts:
            cv2.circle(im, (int(x), int(y)), 15, (0, 0, 255), 2)
    cv2.imencode(".png", im)[1].tofile(path)


# ---------------------------------------------------------------- 主流程
def natural_key(name):
    """A1, A1', A2, A2', ... 的自然顺序。"""
    stem = os.path.splitext(name)[0]
    m = re.match(r"^([A-Za-z]+)(\d+)(\'?)$", stem)
    if m:
        return (m.group(1).lower(), int(m.group(2)), 1 if m.group(3) else 0)
    return (stem.lower(), -1, 0)


def find_images(folder):
    files = []
    for name in os.listdir(folder):
        if name.startswith("_") or name.startswith("."):
            continue
        if name.lower().endswith(EXTS):
            files.append(name)
    return sorted(files, key=natural_key)


def main():
    parser = argparse.ArgumentParser(
        description="自动统计显微镜照片中的细胞数量，输出 CSV")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input-dir", default=script_dir,
                        help="图片所在目录（默认：脚本所在目录）")
    parser.add_argument("--output", default=None,
                        help="输出 CSV 路径（默认：<图片目录>/cell_counts.csv）")
    parser.add_argument("--overlays", action="store_true",
                        help="同时保存带标注的预览图到 cell_count_overlays/ 目录")
    args = parser.parse_args()

    folder = os.path.abspath(args.input_dir)
    output = args.output or os.path.join(folder, "cell_counts.csv")
    files = find_images(folder)
    if not files:
        print("目录中没有找到图片:", folder)
        return 1

    overlay_dir = os.path.join(folder, "cell_count_overlays")
    if args.overlays:
        os.makedirs(overlay_dir, exist_ok=True)

    rows = []
    for idx, name in enumerate(files, 1):
        path = os.path.join(folder, name)
        try:
            img = read_image(path)
        except Exception as e:
            print("[%d/%d] %s  读取失败: %s" % (idx, len(files), name, e))
            continue

        modality = detect_modality(img)
        if modality == "fluorescence":
            result = count_fluorescence(img)
            count = result[0]
        else:
            result = count_brightfield(img)
            count = result[0]

        rows.append((name, modality, count))
        print("[%d/%d] %-14s %-13s %d" % (idx, len(files), name, modality, count),
              flush=True)

        if args.overlays:
            out_name = os.path.splitext(name)[0].replace("'", "p") + "_counted.png"
            save_overlay(img, modality, result, os.path.join(overlay_dir, out_name))

    with open(output, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "modality", "cell_count"])
        writer.writerows(rows)

    print("\nProcessed %d images. Results written to:\n%s" % (len(rows), output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
