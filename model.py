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
        assert stride in [1, 2]
        self.stride = stride
        hidden_dim = round(input_channels * expand_ratio)
        self.ShortCut = True if self.stride == 1 else False
        self.add_conv = True if input_channels != output_channels else False
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 1, bias = False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace = False),

            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace = False),

            nn.Conv2d(hidden_dim, output_channels, 1, bias = False),
            nn.BatchNorm2d(output_channels)
        )
        if stride == 1 and self.add_conv:
            self.res_conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, bias = False),
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
        if self.ShortCut:
            if self.add_conv:
                return conv + self.res_conv(input)
            return input + conv
        return conv


class MobileNetV2(nn.Module):
    def __init__(self, num_class, width_multiplier = 1.):
        """
        Introduction
        ------------
            MobileNetV2模型构造函数
        Parameters
        ----------
            num_class: 数据集类别数量
            width_multiplier: 宽度因子
        """
        super(MobileNetV2, self).__init__()
        self.num_class = num_class
        self.InvertedResidualSettings = [
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
        last_channels = int(1280 * width_multiplier) if width_multiplier > 1.0 else 1280

        self.net = [nn.Sequential(
            nn.Conv2d(3, input_channels, 3, stride = 2, padding = 1,  bias = False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace = False)
        )]

        for t, c, n, s in self.InvertedResidualSettings:
            output_channels = int(c * width_multiplier)
            for i in range(n):
                # 只有第一层的网络stride为2
                if i == 0:
                    self.net.append(InvertedResidual(input_channels, output_channels, stride = s, expand_ratio = t))
                else:
                    self.net.append(InvertedResidual(input_channels, output_channels, stride = 1, expand_ratio = t))
                input_channels = output_channels
        self.net.append(nn.Sequential(
            nn.Conv2d(input_channels, last_channels, 1, stride = 1, bias = False),
            nn.BatchNorm2d(last_channels),
            nn.ReLU6(inplace = False)
        ))

        self.net.append(nn.AvgPool2d(kernel_size = 7))
        self.net.append(nn.Conv2d(last_channels, self.num_class, 1, bias = True))

        self.net = nn.Sequential(*self.net)


    def forward(self, input):
        """
        Introduction
        ------------
            前向传播计算
        Parameters
        ----------
            input: 输入特征变量
        """
        conv = self.net(input)
        conv = conv.view(-1, self.num_class)
        return conv


    def initialize(self):
        """
        Introduction
        ------------
            初始化模型参数
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

