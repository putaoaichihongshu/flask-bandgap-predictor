import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from torch.optim import LBFGS


# 定义 e^x 激活函数
class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)


# 定义 X_net
class NetX(nn.Module):
    def __init__(self):
        super(NetX, self).__init__()
        self.layer = nn.Linear(3, 1)  # 3个输入（Br, Cl, I），1个输出
        self.activation = ExpActivation()

        # 初始化权重为正值
        with torch.no_grad():
            # 使用均匀分布初始化权重为正值 (0, 1)
            nn.init.uniform_(self.layer.weight, 0, 1)
            # 或者使用正态分布的绝对值
            # self.layer.weight.data = torch.abs(torch.randn_like(self.layer.weight))
            nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x


# 组合网络
class FullNet(nn.Module):
    def __init__(self):
        super(FullNet, self).__init__()
        self.netX = NetX()

        # 组合网络：3(MA,FA,Cs) + 1(X_net输出) = 4个输入
        self.combination_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, site_a, site_x):
        output_x = self.netX(site_x)
        combined_features = torch.cat([site_a, output_x], dim=1)
        final_output = self.combination_net(combined_features)
        return final_output
