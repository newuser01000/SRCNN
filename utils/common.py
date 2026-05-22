import rasterio
import torch
import torchvision.io as io
import torchvision.transforms as transforms
import numpy as np
import os


def rgb2ycbcr(src):
    R = src[0]
    G = src[1]
    B = src[2]

    y = 0.299 * R + 0.587 * G + 0.114 * B
    return y.unsqueeze(0) # 返回单亮度通道


def sorted_list(dir):
    ls = os.listdir(dir)
    ls.sort()
    for i in range(0, len(ls)):
        ls[i] = os.path.join(dir, ls[i])
    return ls

def resize_bicubic(src, h, w):
    image = transforms.Resize((h, w), transforms.InterpolationMode.BICUBIC)(src)
    return image

def gaussian_blur(src, ksize=3, sigma=0.5):
    # 图像模糊算子（高斯核大小，模糊强度）（模糊图像）
    blur_image = transforms.GaussianBlur(kernel_size=ksize, sigma=sigma)(src)
    return blur_image
    
def upscale(src, scale):
    h = int(src.shape[1] * scale)
    w = int(src.shape[2] * scale)
    image = resize_bicubic(src, h, w)
    return image

def downscale(src, scale):
    h = int(src.shape[1] / scale)
    w = int(src.shape[2] / scale)
    image = resize_bicubic(src, h, w) #双三次插值缩小
    return image

def make_lr(src, scale=2):
    h = src.shape[1]  # H
    w = src.shape[2]  # W
    lr_image = downscale(src, scale) # 缩小
    lr_image = resize_bicubic(lr_image, h, w) # 双三次插值放大
    return lr_image

def exists(path):
    return os.path.exists(path) # 判断路径是否存在

def PSNR(y_true, y_pred, max_val=1):
    y_true = y_true.type(torch.float32)
    y_pred = y_pred.type(torch.float32)
    MSE = torch.mean(torch.square(y_true - y_pred))
    return 10 * torch.log10(max_val * max_val / MSE)

def random_crop(src, h, w):
    crop = transforms.RandomCrop([h, w])(src)
    return crop

def random_transform(src):
    # 旋转、翻转
    operations = {
        0 : (lambda x : x                                       ),
        1 : (lambda x : torch.rot90(x, k=1,  dims=(1, 2))),
        2 : (lambda x : torch.rot90(x, k=2,  dims=(1, 2))),
        3 : (lambda x : torch.rot90(x, k=3,  dims=(1, 2))),
        4 : (lambda x : torch.fliplr(x, axis=2)                 ),
        5 : (lambda x : torch.flipud(x, axis=1)                 ),
    }
    idx = np.random.choice([0, 1, 2, 3, 4, 5])
    image_transform = operations[idx](src).copy() # 对[H, W]进行操作
    return image_transform

def shuffle(X, Y):
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of elements")
    indices = np.arange(0, X.shape[0])  # 生成索引数组
    np.random.shuffle(indices)          # 随机打乱索引顺序
    X = torch.index_select(X, dim=0, index=torch.as_tensor(indices))  # 按新的随机索引重新排列
    Y = torch.index_select(Y, dim=0, index=torch.as_tensor(indices))
    return X, Y

def tensor2numpy(tensor):
    return tensor.detach().cpu().numpy()

def to_cpu(tensor):
    return tensor.detach().cpu()

def read_tif(image_path):
    with rasterio.open(image_path) as src:
        #读取B8，B2，B3，B4波段并转化为32位浮点数
        img = src.read([1, 2, 3, 4]).astype(np.float32)
    #统一归一化至（0， 1）
    img = np.clip(img, 0, 10000) / 10000.0
    return img
