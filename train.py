from utils.dataset import dataset
from utils.common import PSNR
from model import SRCNN
import argparse
import torch
import os

parser = argparse.ArgumentParser() # 创建参数收集器，收集终端输入的参数
parser.add_argument("--steps",          type=int, default=100000,   help='-') # 更新参数次数
parser.add_argument("--batch-size",     type=int, default=128,      help='-') # 批大小
parser.add_argument("--architecture",   type=str, default="915",    help='-') # 架构
parser.add_argument("--save-every",     type=int, default=1000,     help='-') # 保存间隔
parser.add_argument("--save-log",       type=int, default=0,        help='-') # 是否保存日志
parser.add_argument("--save-best-only", type=int, default=0,        help='-') # 是否只保存最佳模型
parser.add_argument("--ckpt-dir",       type=str, default="",       help='-') # 模型保存路径
parser.add_argument("--scale",          type=int, default=2,      help='-') # 训练倍率
parser.add_argument("--mode",           type=str, default="B234",help='-') # 训练模式

# 从终端识别参数
FLAGS, unparsed = parser.parse_known_args() # 识别已知参数，忽略未知参数
steps      = FLAGS.steps
batch_size = FLAGS.batch_size
save_every = FLAGS.save_every
save_log   = (FLAGS.save_log == 1) # save-log为1时开启日志保存
ckpt_dir   = FLAGS.ckpt_dir
scale      = FLAGS.scale
mode       = FLAGS.mode
save_best_only = (FLAGS.save_best_only == 1) # 为1时只保存最佳模型
architecture   = FLAGS.architecture

if architecture not in ["915", "935", "955"]:
    raise ValueError("architecture must be 915, 935, 955")

if (ckpt_dir == "") or (ckpt_dir == "default"):
    ckpt_dir = f"checkpoint/SRCNN{architecture}/x{scale}/AdamW/{mode}" # 模型保存路径------------------------------------------------------
os.makedirs(ckpt_dir, exist_ok=True)

model_path = os.path.join(ckpt_dir, f"{mode}SRCNN-{architecture}.pt") # (最佳)模型权重文件
ckpt_path = os.path.join(ckpt_dir, f"{mode}ckpt.pt") # 检查点文件

if mode == "B234":
    channels = 3
elif mode == "B8":
    channels = 1
else:
    raise ValueError("mode must be B234 or B8")

# 初始化参数
dataset_dir = "dataset" # 数据集路径
lr_crop_size = 33 # 低分辨率裁剪尺寸
hr_crop_size = 21 # 高分辨率裁剪尺寸
if architecture == "935":
    hr_crop_size = 19
elif architecture == "955":
    hr_crop_size = 17

# 创建训练集对象
train_set = dataset(dataset_dir, "train", scale, mode, architecture)
train_set.generate(lr_crop_size, hr_crop_size)
train_set.load_data()

# 创建验证集对象
valid_set = dataset(dataset_dir, "validation", scale, mode, architecture)
valid_set.generate(lr_crop_size, hr_crop_size)
valid_set.load_data()

data, labels, isEnd = train_set.get_batch(128, shuffle_each_epoch=False)  # 获取训练输入、标签

# 开始训练
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    srcnn = SRCNN(architecture, device, channels)
    srcnn.setup(optimizer=torch.optim.AdamW(srcnn.model.parameters(), lr=2e-5, weight_decay=1e-5),  # 优化器
                loss=torch.nn.MSELoss(), # 损失函数
                model_path=model_path,   # 模型保存路径
                ckpt_path=ckpt_path,     # 中间训练状态
                metric=PSNR)             # 评价指标

    srcnn.load_checkpoint(ckpt_path)     # 从断点训练恢复
    srcnn.train(train_set, valid_set,
                steps=steps,
                batch_size=batch_size,
                save_best_only=save_best_only,

                save_every=save_every,
                save_log=save_log,
                log_dir=ckpt_dir)

if __name__ == "__main__":
    main()
