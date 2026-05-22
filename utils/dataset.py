from utils.common import *
import numpy as np
import torch
import os
import torch.nn.functional as F

class dataset:
    def __init__(self, dataset_dir, subset, scale, mode, architecture="915"):
        # 保存为内部变量
        self.dataset_dir = dataset_dir  # 指定数据集位置
        self.subset = subset            # 指定数据子集
        self.data = torch.Tensor([])    # 创建空张量保存数据
        self.labels = torch.Tensor([])  # 创建空标签张量
        self.architecture = architecture
        self.scale= scale
        self.data_file = os.path.join(self.dataset_dir, f"x{scale}/{architecture}/{mode}_data_{self.subset}.npy")        # 输入数据文件路径
        self.labels_file = os.path.join(self.dataset_dir, f"x{scale}/{architecture}/{mode}_labels_{self.subset}.npy")    # 标签文件路径
        self.cur_idx = 0                # 当前读取索引
        self.mode = mode


    def generate(self, lr_crop_size, hr_crop_size, transform=False):

        # 检查文件是否存在
        if exists(self.data_file) and exists(self.labels_file):  # dataset/x2
            print(f"{self.data_file} and {self.labels_file} HAVE ALREADY EXISTED\n")
            return

        data, labels = [], []

        # 图像对准，LR
        padding = np.absolute(hr_crop_size - lr_crop_size) // 2
        step = 14

        train_data = os.path.join(self.dataset_dir, self.subset, "data")      # 训练数据路径
        train_labels = os.path.join(self.dataset_dir, self.subset, "labels")  # 训练标签路径

        # 文件排序
        dt_img = sorted_list(train_data)
        lb_img = sorted_list(train_labels)

        if len(lb_img) == 0:
            raise ValueError(f"{train_labels} 文件夹为空，无法生成数据集。")

        if self.mode == "B234":
            if len(dt_img) == 0:
                raise ValueError(f"{train_data} 文件夹为空，无法生成 B234 数据集。")
            if len(dt_img) != len(lb_img):
                raise ValueError(
                    f"data 和 labels 图像数量不一致：\n"
                    f"data 数量 = {len(dt_img)}\n"
                    f"labels 数量 = {len(lb_img)}"
                )
            img_pairs = list(zip(dt_img, lb_img))
        elif self.mode == "B8":
            # B8 模式不使用 data 中的 B8A，而是从 HR_B8 退化生成 LR_B8
            img_pairs = [(None, lb_path) for lb_path in lb_img]
        else:
            raise ValueError("mode must be B234 or B8")

        for dt_path, lb_path in img_pairs:
            # zip()将可迭代的对象打包成一个元组，并返回列表[(dt_img[0],lb_img[0]), (dt_img[1],lb_img),...]
            # 读取标签所有波段
            hr_image = read_tif(lb_path)

            if hr_image.shape[0] < 4:
                raise ValueError(
                    f"HR 影像波段数量不足 4 个：\n"
                    f"path = {lb_path}\n"
                    f"shape = {hr_image.shape}\n"
                    f"请确认波段顺序为 B8, B2, B3, B4。"
                )


            if self.mode == "B234":
                lr_image = read_tif(dt_path)
                if lr_image.shape != hr_image.shape:
                    raise ValueError(
                        f"LR 和 HR 尺寸不一致：\n"
                        f"LR shape = {lr_image.shape}\n"
                        f"HR shape = {hr_image.shape}\n"
                    )
                # 合成后的波段顺序，近红外、红、绿、蓝
                hr_image = hr_image[[1, 2, 3], :, :]
                lr_image = lr_image[[1, 2, 3], :, :]

            elif self.mode == "B8":
                hr_image = hr_image[[0], :, :]
                lr_image = self.mk_lr(hr_image)

            if lr_image.max() == 0 or hr_image.max() == 0: continue

            if transform: # 是否数据增强
                seed = np.random.randint(0, 100000)

                np.random.seed(seed)
                lr_image = random_transform(lr_image)
                np.random.seed(seed)
                hr_image = random_transform(hr_image)

            h = hr_image.shape[1]
            w = hr_image.shape[2]

            # 双窗口平滑
            for x in np.arange(start=0, stop=h - lr_crop_size, step=step):
                for y in np.arange(start=0, stop=w - lr_crop_size, step=step):

                    subim_data = lr_image[
                                 :,
                                 x: x + lr_crop_size,
                                 y: y + lr_crop_size
                                 ]

                    subim_label = hr_image[
                                  :,
                                  x + padding: x + padding + hr_crop_size,
                                  y + padding: y + padding + hr_crop_size
                                  ]
                    # 过滤全零patch
                    if subim_data.max() == 0 or subim_label.max() == 0:
                        continue

                    valid_mask = np.any(subim_label > 0, axis=0)
                    valid_ratio = valid_mask.mean()

                    if valid_ratio < 0.6:
                        continue

                    data.append(subim_data)
                    labels.append(subim_label)

        if len(data) == 0:
            raise ValueError(
                f"{self.mode} {self.subset} 没有生成任何样本。\n"
                f"请检查影像路径、波段顺序、有效像素比例阈值 valid_threshold。"
            )

        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        np.save(self.data_file, data)
        np.save(self.labels_file, labels)

        print(f"{self.mode} {self.subset} 数据集生成完成！")
        print("data shape:", data.shape)
        print("labels shape:", labels.shape)
        print("LR range:", data.min(), data.max())
        print("HR range:", labels.min(), labels.max())
        print("data >= 1 ratio:", np.mean(data >= 1.0))
        print("labels >= 1 ratio:", np.mean(labels >= 1.0))
        print("data <= 0 ratio:", np.mean(data <= 0.0))
        print("labels <= 0 ratio:", np.mean(labels <= 0.0))

    def load_data(self):
        # 检查数据集是否存在
        if not exists(self.data_file):
            raise ValueError(f"\n{self.data_file} and {self.labels_file} DO NOT EXIST\n")
        self.data = np.load(self.data_file)      # 从磁盘读取文件
        self.data = torch.as_tensor(self.data)   # 转换为张量
        self.labels = np.load(self.labels_file)
        self.labels = torch.as_tensor(self.labels)

    def get_batch(self, batch_size, shuffle_each_epoch=True):
        isEnd = False  # 该批次是否达到本轮末尾
        if self.cur_idx + batch_size > self.data.shape[0]:
            isEnd = True
            self.cur_idx = 0
            if shuffle_each_epoch:
                self.data, self.labels = shuffle(self.data, self.labels) # 打乱顺序

        data = self.data[self.cur_idx : self.cur_idx + batch_size]
        labels = self.labels[self.cur_idx : self.cur_idx + batch_size]
        self.cur_idx += batch_size

        return data, labels, isEnd

    def mk_lr(self, hr_image):
        # 转换成32位浮点数组
        hr_image = np.asarray(hr_image, dtype=np.float32)
        c, h ,w = hr_image.shape

        h_lr = h // self.scale
        w_lr = w // self.scale

        # 转化位torch Tensor，并增加一个维度[N, C, H, W]
        x = torch.from_numpy(hr_image).unsqueeze(0)

        # 模拟退化过程
        x_lr = F.interpolate(
            x,
            size=(h_lr, w_lr),
            mode="bicubic",
            align_corners=False
        )

        x_up = F.interpolate(
            x_lr,
            size=(h, w),
            mode="bicubic",
            align_corners=False
        )

        # 去掉第零维，转化回数组
        lr_image = x_up.squeeze(0).numpy().astype(np.float32)
        if hr_image.min() >= 0 and hr_image.max() <= 1.0:
            lr_image = np.clip(lr_image, 0.0, 1.0)

        return lr_image