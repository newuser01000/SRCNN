import argparse
from rasterio.transform import Affine
import rasterio
from model import SRCNN
from utils.common import *

parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=int,   default=2,                   help='-') # 超分倍率
parser.add_argument("--b234-ckpt-path",    type=str,   default="",             help='-') # B234模型路径
parser.add_argument("--b8-ckpt-path",      type=str,   default="",             help='-') # B8模型路径
parser.add_argument('--architecture', type=str,   default="915",               help='-') # 模型参数
parser.add_argument("--image-path",   type=str,   default="D:/Microsoft Edge/yinhaiqu/LR/Yinhai.tif",   help='-') # 超分图像路径
parser.add_argument("--out_sr",       type=str,   default="",         help='-') # 超分结果输出路径
parser.add_argument("--out_bicubic",  type=str,   default="",         help='-')

# 从终端读取参数
FLAGS, unparsed = parser.parse_known_args()

image_path = FLAGS.image_path
architecture = FLAGS.architecture
scale = FLAGS.scale
B234_ckpt_path = FLAGS.b234_ckpt_path
B8_ckpt_path = FLAGS.b8_ckpt_path
out_sr = FLAGS.out_sr
out_bicubic = FLAGS.out_bicubic

if architecture not in ["915", "935", "955"]:
    raise ValueError("architecture must be 915, 935, 955")
if scale not in [2, 3, 4]:
    raise ValueError("must be 2, 3 or 4")
if (B234_ckpt_path == "") or (B234_ckpt_path == "default"):
    B234_ckpt_path = f"checkpoint/SRCNN{architecture}/x{scale}/AdamW/B234/B234SRCNN-{architecture}.pt"
if (B8_ckpt_path == "") or (B8_ckpt_path == "default"):
    B8_ckpt_path = f"checkpoint/SRCNN{architecture}/x{scale}/AdamW/B8/B8SRCNN-{architecture}.pt"
if out_sr == "":
    out_sr = f"dataset/x{scale}/{architecture}/SR_{architecture}.tif"
if out_bicubic == "":
    out_bicubic = f"dataset/x{scale}/{architecture}/BC_{architecture}.tif"

sigma = 0.3 if scale == 2 else 0.2 #
pad = int(architecture[1]) // 2 + 6


def read_geotiff(img_path):
    with rasterio.open(img_path) as src:
        data = src.read([1, 2, 3, 4])
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    original_dtype = data.dtype

    data = data.astype(np.float32)
    data = data / 10000.0
    data = np.clip(data, 0.0, 1.0)

    img = torch.from_numpy(data).float()

    return img, profile, transform, crs, original_dtype

def save_geotiff(path, img_tensor, profile, transform, crs, original_dtype):

    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    img = img_tensor.cpu().numpy().astype(np.float32)

    out = img * 10000.0
    dtype = np.dtype(original_dtype)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(out, info.min, info.max)

    out = out.astype(dtype)

    out_profile = profile.copy()
    out_profile.update({
        "driver": "GTiff",
        "height": out.shape[1],
        "width": out.shape[2],
        "count": 4,
        "dtype": str(dtype),
        "crs": crs,
        "transform": transform,
        "interleave": "band"
    })

    # 防止原 profile 中残留不适合的字段
    out_profile.pop("photometric", None)

    out_dir = os.path.dirname(path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out)


def crop_valid_region(img):
    # img: [C,H,W]
    valid = torch.any(img > 0, dim=0)
    rows = torch.where(valid.any(dim=1))[0]
    cols = torch.where(valid.any(dim=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return img, 0, 0

    r0, r1 = rows[0].item(), rows[-1].item() + 1
    c0, c1 = cols[0].item(), cols[-1].item() + 1

    return img[:, r0:r1, c0:c1], r0, c0

def auto_crop_black_border(img, thresh=1e-6):

    # 只要任一通道大于阈值，就认为该像素有效
    valid = torch.any(img > thresh, dim=0)   # [H,W]

    rows = torch.where(valid.any(dim=1))[0]
    cols = torch.where(valid.any(dim=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return img, 0, 0

    top = rows[0].item()
    bottom = rows[-1].item() + 1
    left = cols[0].item()
    right = cols[-1].item() + 1

    cropped = img[:, top:bottom, left:right]
    return cropped, left, top

def sr_one_branch(model, lr_image, scale, device):

    # 1. 双三次上采样
    bicubic_image = upscale(lr_image, scale)  # [C, H*scale, W*scale]

    input_tensor = bicubic_image.unsqueeze(0).to(device)  # [1, C, H, W]

    # 2. SRCNN预测
    model.model.eval()
    with torch.no_grad():
        sr_patch = model.predict(input_tensor)[0].cpu()  # [C, H_out, W_out]

    # 3. 用 SRCNN 输出替换 bicubic 中心区域
    _, h_in, w_in = bicubic_image.shape
    _, h_out, w_out = sr_patch.shape

    crop_top = (h_in - h_out) // 2
    crop_left = (w_in - w_out) // 2

    sr_image = bicubic_image.clone()

    sr_image[
        :,
        crop_top:crop_top + h_out,
        crop_left:crop_left + w_out
    ] = sr_patch

    sr_image = torch.clamp(sr_image, 0.0, 1.0)
    bicubic_image = torch.clamp(bicubic_image, 0.0, 1.0)
    return sr_image, bicubic_image

#----------------------------
#------------------------------
#--------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_234 = SRCNN(architecture, device, 3)
    model_234.load_weights(B234_ckpt_path)
    model_8 = SRCNN(architecture, device, 1)
    model_8.load_weights(B8_ckpt_path)

    # 1. 读取带坐标的LR tif
    lr_image, profile, transform, crs, original_dtype = read_geotiff(image_path)

    # 如果是全零图像，直接跳过
    if lr_image.max().item() == 0:
        print("输入影像全零，跳过")
        return

    # 裁掉无效黑边
    lr_image, r0, c0 = crop_valid_region(lr_image)
    transform = transform * Affine.translation(c0, r0)

    lr_b8 = lr_image[[0], :, :]
    lr_b234 = lr_image[[1, 2, 3], :, :]

    sr_234, bicubic_234 = sr_one_branch(model_234, lr_b234, scale, device)
    sr_8, bicubic_8 = sr_one_branch(model_8, lr_b8, scale, device)

    if sr_234.shape[1:] != sr_8.shape[1:]:
        raise ValueError(
            f"B234 和 B8 超分结果尺寸不一致：\n"
            f"SR_B234 shape = {sr_234.shape}\n"
            f"SR_B8 shape = {sr_8.shape}"
        )
    if bicubic_234.shape[1:] != bicubic_8.shape[1:]:
        raise ValueError(
            f"B234 和 B8 双三次结果尺寸不一致：\n"
            f"Bicubic_B234 shape = {bicubic_234.shape}\n"
            f"Bicubic_B8 shape = {bicubic_8.shape}"
        )

    sr = torch.cat([sr_8, sr_234], dim=0)
    bicubic = torch.cat([bicubic_8, bicubic_234], dim=0)

    # 6. 更新空间分辨率
    sr_transform = transform * Affine.scale(1 / scale, 1 / scale)
    bicubic_transform = sr_transform

    # 7. 去除输出黑边
    sr, left, top = auto_crop_black_border(sr, thresh=1e-6)

    bicubic = bicubic[:, top:top + sr.shape[1], left:left + sr.shape[2]]

    sr_transform = sr_transform * Affine.translation(left, top)
    bicubic_transform = bicubic_transform * Affine.translation(left, top)

    # 8. 保存 B234 超分结果
    save_geotiff(
        out_sr,
        sr,
        profile,
        sr_transform,
        crs,
        original_dtype
    )
    save_geotiff(
        out_bicubic,
        bicubic,
        profile,
        bicubic_transform,
        crs,
        original_dtype
    )

    print("SRCNN结果保存到:", out_sr)
    print("Bicubic结果保存到:", out_bicubic)


if __name__ == "__main__":
    main()