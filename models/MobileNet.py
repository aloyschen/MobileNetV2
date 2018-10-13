import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, input_channels, output_channels, stride, expand_ratio):
        """
        Introduction
        ------------
            构造函数
        Parameters
        ----------
            input_channels: 输入的feature map的channel数量
            output_channels: 输出的feature map的channel数量
            stride: 卷积步长
            expand_ratio: inverted Residual结构扩充通道的倍数
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(input_channels * expand_ratio)
        self.use_res_connect = stride == 1 and input_channels == output_channels
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 1, bias = False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),

            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),

            nn.Conv2d(hidden_dim, output_channels, 1, bias = False),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, input):
        """
        Introduction
        ------------
            前向计算结构
        Parameters
        ----------
            input: 输入变量
        Returns
        -------
            conv: 经过Inverted Residual block计算得到的feature map
        """
        conv = self.block(input)
        if self.use_res_connect:
            return input + conv
        return conv

class _conv_bn(nn.Module):
    """
    Introduciton
    ------------
        MobileNet V2 基本卷积结构
    """
    def __init__(self, inp, oup, stride):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)

def MobileNetV2(width_multiplier = 1.):
    """
    Introduction
    ------------
        MobileNetV2模型构造函数
    Parameters
    ----------
        num_class: 数据集类别数量
        width_multiplier: 宽度因子
    """
    layers = []
    InvertedResidualSettings = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]
    input_channels = int(32 * width_multiplier)
    layers += [_conv_bn(3, input_channels, stride = 2)]
    for t, c, n, s in InvertedResidualSettings:
        output_channels = int(c * width_multiplier)
        for i in range(n):
            # 只有第一层的网络stride为2
            if i == 0:
                layers += [InvertedResidual(input_channels, output_channels, stride = s, expand_ratio = t)]
            else:
                layers += [InvertedResidual(input_channels, output_channels, stride = 1, expand_ratio = t)]
            input_channels = output_channels

    return layers
