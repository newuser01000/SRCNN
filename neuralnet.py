import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN_model(nn.Module):
    def __init__(self, architecture : str, channels = 3 ) -> None:
        super().__init__()  # 调用父类初始化

        if architecture not in ["915", "935", "955"]:
            raise ValueError("architecture must be 915, 935 or 955")
        k = int(architecture[1])

        self.patch_extraction = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=9)  # 定义卷积层
        nn.init.normal_(self.patch_extraction.weight, mean=0.0, std=0.001)                # 初始化权值
        nn.init.zeros_(self.patch_extraction.bias)                                        # 初始化偏执

        self.nonlinear_map = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=k)
        nn.init.normal_(self.nonlinear_map.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.nonlinear_map.bias)

        self.recon = nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=5)
        nn.init.normal_(self.recon.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.recon.bias)

    def forward(self, X_in):
        X = F.relu(self.patch_extraction(X_in))  # 经过第一层卷积后ReLu激活
        X = F.relu(self.nonlinear_map(X))        # 第二层卷积
        X = self.recon(X)                        # 第三次卷积
        X_out = torch.clip(X, 0.0, 1.0) # 限制输出像素范围
        return X_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
