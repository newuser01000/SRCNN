import os
import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import torch
import torch.nn.functional as F


parser = argparse.ArgumentParser()

parser.add_argument("--hr",      type=str, default="D:/Microsoft Edge/yinhaiqu/HR/Yinhai.tif",           help="真实原始HR影像路径")

parser.add_argument("--sr",      type=str, default="dataset/x2/915/SR.tif",         help="SRCNN超分重建结果路径")

parser.add_argument("--bicubic", type=str, default="dataset/x2/915/BC.tif",    help="双三次插值结果路径")

FLAGS = parser.parse_args()

hr_path = FLAGS.hr
sr_path = FLAGS.sr
bicubic_path = FLAGS.bicubic


BAND_NAMES = ["B8", "B4", "B3", "B2"]

# def band_stats(path, name):
#     with rasterio.open(path) as src:
#         img = src.read([1, 2, 3, 4]).astype(np.float32)
#
#     print(f"\n{name}")
#     print("Band order: B8, B4, B3, B2")
#
#     band_names = ["B8", "B4", "B3", "B2"]
#
#     for i, band_name in enumerate(band_names):
#         band = img[i]
#         valid = band > 0
#
#         print(
#             f"{band_name}: "
#             f"mean={band[valid].mean():.4f}, "
#             f"std={band[valid].std():.4f}, "
#             f"min={band[valid].min():.4f}, "
#             f"max={band[valid].max():.4f}"
#         )
#
#
# band_stats(hr_path, "HR")
# band_stats(sr_path, "SRCNN")
# band_stats(bicubic_path, "Bicubic")

def normalize_to_01(img):

    img = img.astype(np.float32)

    max_val = np.nanmax(img)

    if max_val > 1.5:
        img = np.clip(img, 0, 10000) / 10000.0
    else:
        img = np.clip(img, 0.0, 1.0)

    return img.astype(np.float32)


def read_reference(path):

    with rasterio.open(path) as src:
        img = src.read([1, 2, 3, 4])
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width

    img = normalize_to_01(img)

    return img, profile, transform, crs, height, width


def read_and_align_to_reference(path, ref_profile, ref_transform, ref_crs, ref_height, ref_width):

    aligned = np.zeros((4, ref_height, ref_width), dtype=np.float32)

    with rasterio.open(path) as src:
        src_data = src.read([1, 2, 3, 4]).astype(np.float32)

        for b in range(4):
            reproject(
                source=src_data[b],
                destination=aligned[b],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )

    aligned = normalize_to_01(aligned)

    return aligned


def calc_rmse(hr, pred, mask):
    diff = hr[mask] - pred[mask]
    return np.sqrt(np.mean(diff ** 2))


def calc_psnr(hr, pred, mask, data_range=1.0):
    rmse = calc_rmse(hr, pred, mask)

    if rmse == 0:
        return float("inf")

    return 20 * np.log10(data_range / rmse)


def calc_masked_ssim(hr, pred, mask, data_range=1.0, win_size=7):

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return np.nan

    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1

    hr_crop = hr[r0:r1, c0:c1]
    pred_crop = pred[r0:r1, c0:c1]
    mask_crop = mask[r0:r1, c0:c1]

    h, w = hr_crop.shape

    if min(h, w) < win_size:
        return np.nan

    x = torch.from_numpy(hr_crop.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    y = torch.from_numpy(pred_crop.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    pad = win_size // 2

    # 使用 reflect padding，减小边缘影响
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y_pad = F.pad(y, (pad, pad, pad, pad), mode="reflect")

    mu_x = F.avg_pool2d(x_pad, kernel_size=win_size, stride=1)
    mu_y = F.avg_pool2d(y_pad, kernel_size=win_size, stride=1)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(x_pad * x_pad, kernel_size=win_size, stride=1) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y_pad * y_pad, kernel_size=win_size, stride=1) - mu_y_sq
    sigma_xy = F.avg_pool2d(x_pad * y_pad, kernel_size=win_size, stride=1) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = (
        (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )

    ssim_map = ssim_map.squeeze().numpy()

    return float(np.mean(ssim_map[mask_crop]))

def calc_visible_sam(hr_img, pred_img, visible_indices=[1, 2, 3], eps=1e-8):
    """
    计算可见光波段 SAM。

    当前影像波段顺序：
        Band 1 = B8
        Band 2 = B4
        Band 3 = B3
        Band 4 = B2

    因此可见光波段为：
        B4, B3, B2 -> index [1, 2, 3]

    hr_img:   [4, H, W]
    pred_img: [4, H, W]

    返回：
        平均 SAM，单位为 degree
    """

    hr_vis = hr_img[visible_indices, :, :]      # [3, H, W]
    pred_vis = pred_img[visible_indices, :, :]  # [3, H, W]

    # 有效区域：HR 和预测影像的可见光波段都大于 0
    valid_mask = np.all(hr_vis > 0, axis=0) & np.all(pred_vis > 0, axis=0)

    # 光谱向量点积
    dot = np.sum(hr_vis * pred_vis, axis=0)

    # 光谱向量模长
    norm_hr = np.sqrt(np.sum(hr_vis ** 2, axis=0))
    norm_pred = np.sqrt(np.sum(pred_vis ** 2, axis=0))

    valid_mask &= (norm_hr > eps) & (norm_pred > eps)

    if np.sum(valid_mask) == 0:
        return np.nan

    cos_angle = dot / (norm_hr * norm_pred + eps)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.arccos(cos_angle)

    # 弧度转角度
    angle_deg = angle * 180.0 / np.pi

    mean_sam = np.mean(angle_deg[valid_mask])

    return mean_sam


def evaluate_one_image_pair(hr_img, pred_img, name):

    results = []

    print(f"\n========== {name} vs HR ==========")

    for i, band_name in enumerate(BAND_NAMES):
        hr_band = hr_img[i]
        pred_band = pred_img[i]

        # 有效区域：真实影像和预测影像都大于0
        mask = (hr_band > 0) & (pred_band > 0)

        valid_count = np.sum(mask)

        if valid_count == 0:
            psnr_value = np.nan
            ssim_value = np.nan
            rmse_value = np.nan
        else:
            rmse_value = calc_rmse(hr_band, pred_band, mask)
            psnr_value = calc_psnr(hr_band, pred_band, mask, data_range=1.0)
            ssim_value = calc_masked_ssim(hr_band, pred_band, mask, data_range=1.0)

        print(
            f"{band_name}: "
            f"PSNR={psnr_value:.4f}, "
            f"SSIM={ssim_value:.4f}, "
            f"RMSE={rmse_value:.6f}, "
        )

        results.append({
            "Method": name,
            "Band": band_name,
            "PSNR": psnr_value,
            "SSIM": ssim_value,
            "RMSE": rmse_value,
        })

    # 计算四波段平均
    avg_psnr = np.nanmean([r["PSNR"] for r in results])
    avg_ssim = np.nanmean([r["SSIM"] for r in results])
    avg_rmse = np.nanmean([r["RMSE"] for r in results])

    visible_sam = calc_visible_sam(
        hr_img=hr_img,
        pred_img=pred_img,
        visible_indices=[1, 2, 3]
    )

    print(
        f"Average: "
        f"PSNR={avg_psnr:.4f}, "
        f"SSIM={avg_ssim:.4f}, "
        f"RMSE={avg_rmse:.6f}, "
        f"Visible SAM={visible_sam:.4f}°"
    )

    results.append({
        "Method": name,
        "Band": "Average",
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "RMSE": avg_rmse,
        "Visible_SAM": visible_sam,
    })

    return results



def main():
    print("读取真实HR影像...")
    hr_img, hr_profile, hr_transform, hr_crs, hr_height, hr_width = read_reference(hr_path)

    print("读取并对齐SRCNN超分结果...")
    sr_img = read_and_align_to_reference(
        sr_path,
        hr_profile,
        hr_transform,
        hr_crs,
        hr_height,
        hr_width
    )

    print("读取并对齐Bicubic插值结果...")
    bicubic_img = read_and_align_to_reference(
        bicubic_path,
        hr_profile,
        hr_transform,
        hr_crs,
        hr_height,
        hr_width
    )

    print("\nHR shape:", hr_img.shape)
    print("SR shape:", sr_img.shape)
    print("Bicubic shape:", bicubic_img.shape)

    all_results = []

    sr_results = evaluate_one_image_pair(
        hr_img=hr_img,
        pred_img=sr_img,
        name="SRCNN"
    )

    bicubic_results = evaluate_one_image_pair(
        hr_img=hr_img,
        pred_img=bicubic_img,
        name="Bicubic"
    )

    all_results.extend(sr_results)
    all_results.extend(bicubic_results)



if __name__ == "__main__":
    main()

