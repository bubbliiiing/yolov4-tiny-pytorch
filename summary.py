import torch
from torchsummary import summary

from nets.yolo4_tiny import YoloBody

import numpy as np


if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = YoloBody(2).to(device)
    summary(m, input_data=(3, 416, 416))
