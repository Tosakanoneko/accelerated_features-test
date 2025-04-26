import torch
import torch.nn as nn

# 定义一个包含 fine_matcher 模块的网络
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

    def forward(self, x):
        return self.fine_matcher(x)

if __name__ == "__main__":
    # 构造一个随机输入数据，batch_size=10，特征维度为128
    x = torch.randn(10, 128)
    model = MLPNet()
    output = model(x)
    print("输出维度：", output.shape)
