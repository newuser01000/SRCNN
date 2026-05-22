from utils.common import *
from utils.dataset import *
from model import SRCNN
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=int, default=2,         help='-')
parser.add_argument('--architecture', type=str, default="915",     help='-')
parser.add_argument('--ckpt-path',    type=str, default="",        help='-')
parser.add_argument('--mode',         type=str, default="B8", help='-')

# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAGS, unparsed = parser.parse_known_args()
scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3, or 4")

architecture = FLAGS.architecture
if architecture not in ["915", "935", "955"]:
    raise ValueError("architecture must be 915, 935, 955")

mode = FLAGS.mode
if mode not in ["B234", "B8"]:
    raise ValueError("mode must be B234 or B8")

ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"checkpoint/SRCNN{architecture}/x{scale}/AdamW/{mode}/{mode}SRCNN-{architecture}.pt"

if mode == "B8":
    channels = 1
elif mode == "B234":
    channels = 3


sigma = 0.3 if scale == 2 else 0.2
pad = int(architecture[1]) // 2 + 6

lr_crop_size = 33
if architecture == "915":
    hr_crop_size = 21
elif architecture == "935":
    hr_crop_size = 19
elif architecture == "955":
    hr_crop_size = 17

def SAM(hr, sr, eps=1e-8, in_degrees=True):
    """
    计算平均光谱角 SAM。

    hr: [N, C, H, W]
    sr: [N, C, H, W]

    返回：
        平均 SAM，单位默认为 degree
    """

    if hr.shape != sr.shape:
        raise ValueError(f"HR 和 SR 尺寸不一致: HR={hr.shape}, SR={sr.shape}")

    if hr.shape[1] < 2:
        return None

    # 光谱向量点积：[N, H, W]
    dot = torch.sum(hr * sr, dim=1)

    # 光谱向量模长：[N, H, W]
    norm_hr = torch.sqrt(torch.sum(hr ** 2, dim=1))
    norm_sr = torch.sqrt(torch.sum(sr ** 2, dim=1))

    # 剔除全零像素，避免除零
    valid_mask = (norm_hr > eps) & (norm_sr > eps)

    cos_angle = dot / (norm_hr * norm_sr + eps)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    angle = torch.acos(cos_angle)

    if valid_mask.sum() == 0:
        return torch.tensor(float("nan"), device=hr.device)

    angle = angle[valid_mask]

    if in_degrees:
        angle = angle * 180.0 / np.pi

    return torch.mean(angle)

# -----------------------------------------------------------
# test 
# -----------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    test_set = dataset(
        dataset_dir = "dataset",
        architecture = architecture,
        subset = "test",
        scale = scale,
        mode = mode,
    )
    test_set.generate(
        lr_crop_size=lr_crop_size,
        hr_crop_size=hr_crop_size,
        transform=False,
    )
    test_set.load_data()

    model = SRCNN(
        architecture = architecture,
        device = device,
        channels = channels,
    )

    model.load_weights(ckpt_path)
    model.model.eval()

    total_psnr=0.0
    total_loss = 0.0
    total_num = 0
    total_sam = 0.0
    sam_num = 0

    mse_loss = torch.nn.MSELoss(reduction="mean")

    if mode == "B234":
        band_names = ["B2", "B3", "B4"]
        band_psnr_sum = np.zeros(3, dtype=np.float64)
    else:
        band_names = ["B8"]
        band_psnr_sum = np.zeros(1, dtype=np.float64)
    with torch.no_grad():
        n = test_set.data.shape[0]
        batch_size = 64
        for start in range(0, n, batch_size):
            end = min(start + 64, n)

            lr = test_set.data[start:end].to(device)
            hr = test_set.labels[start:end].to(device)

            sr = model.predict(lr)

            loss = mse_loss(sr, hr)
            psnr = PSNR(hr, sr, max_val=1)
            sam = None
            if mode == "B234":
                sam = SAM(hr, sr, in_degrees=True)

            cur_batch = end - start

            total_loss += loss.item() * cur_batch
            total_psnr += psnr.item() * cur_batch
            total_num += cur_batch

            if sam is not None and not torch.isnan(sam):
                total_sam += sam.item() * cur_batch
                sam_num += cur_batch

            # 分波段 PSNR
            for b in range(channels):
                band_psnr = PSNR(
                    hr[:, b:b + 1, :, :],
                    sr[:, b:b + 1, :, :],
                    max_val=1
                )
                band_psnr_sum[b] += band_psnr.item() * cur_batch


        avg_loss = total_loss / total_num
        avg_psnr = total_psnr / total_num
        avg_band_psnr = band_psnr_sum / total_num

        if sam_num > 0:
            avg_sam = total_sam / sam_num
        else:avg_sam = None

        print("\n========== 测试结果 ==========")
        print(f"Mode: {mode}")
        print(f"Architecture: {architecture}")
        print(f"Scale: x{scale}")
        print(f"Average Loss: {avg_loss:.7f}")
        print(f"Average PSNR: {avg_psnr:.4f} dB")

        if avg_sam is not None:
            print(f"Average SAM: {avg_sam:.4f}°")
        else:
            print("Average SAM: 当前为单波段 B8，不计算 SAM")

        print("\n各波段 PSNR:")
        for name, value in zip(band_names, avg_band_psnr):
            print(f"{name}: {value:.4f} dB")

if __name__ == "__main__":
    main()

