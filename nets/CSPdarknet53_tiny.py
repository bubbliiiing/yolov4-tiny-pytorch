import math

import torch
import torch.nn as nn


#-------------------------------------------------#
#   卷积块
#   Conv2d + BatchNorm2d + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

         
'''
                    input
                      |
                  BasicConv
                      -----------------------
                      |                     |
                 route_group              route
                      |                     |
                  BasicConv                 |
                      |                     |
    -------------------                     |
    |                 |                     |
 route_1          BasicConv                 |
    |                 |                     |
    -----------------cat                    |
                      |                     |
        ----      BasicConv                 |
        |             |                     |
      feat           cat---------------------
                      |
                 MaxPooling2D
'''
#---------------------------------------------------#
#   CSPdarknet53-tiny的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
#---------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels//2, out_channels//2, 3)
        self.conv3 = BasicConv(out_channels//2, out_channels//2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2,2],[2,2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim = 1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x,route1], dim = 1) 

        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim = 1)
        
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x,feat

class CSPDarkNet(nn.Module):
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 =  Resblock_body(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 =  Resblock_body(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 =  Resblock_body(256, 256)
        # 13,13,512 -> 13,13,512
        self.conv3 = BasicConv(512, 512, kernel_size=3)

        self.num_features = 1
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _    = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _    = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1    = self.resblock_body3(x)

        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1,feat2

def darknet53_tiny(pretrained, **kwargs):
    model = CSPDarkNet()
    if pretrained:
        model.load_state_dict(torch.load("model_data/CSPdarknet53_tiny_backbone_weights.pth"))
    return model
