import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Modules import L2Norm

class SSDLite(nn.Module):
    def __init__(self, BaseModel, extras, head, feature_layer, num_class):
        """
        Introduction
        ------------
            Single Shot Multibox Architecture Lite
            SSD模型初始化函数，在base model提取特征基础上，增加multi box卷积获取类别概率，box坐标等
        Parameters
        ----------
            BaseModel: vgg基础网络结构
            extras: ssd添加的多尺度特征提取层
            head: multibox中的loc和class conf layer
            num_class: 训练集类别数量
            size: 图片的大小，包括300，500尺寸
            feature_layer: 提取MobileNetv2的多少层提取特征
        """
        super(SSDLite, self).__init__()
        self.num_class = num_class
        self.base = nn.ModuleList(BaseModel)
        self.L2Norm = L2Norm(feature_layer[1][0], 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.feature_layer = feature_layer


    def forward(self, x, training):
        """
        Introduction
        ------------
            模型前向传播结构
        Parameters
        ----------
            x: 输入特征变量
            training: 是否为训练
        """
        sources = []
        loc = []
        conf = []
        # Mobilenet V2模型的特征层使用[19, 19, 96] [10, 10, 320]
        for index in range(len(self.base)):
            x = self.base[index](x)
            if index in self.feature_layer[0]:
                if len(sources) == 0:
                    x = self.L2Norm(x)
                    sources.append(x)
                else:
                    sources.append(x)

        # 添加多尺度特征层[5, 5, 512], [3, 3, 256], [2, 2, 256], [1, 1, 128]
        for index, value in enumerate(self.extras):
            x = F.relu(value(x))
            sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1))
            conf.append(c(x).permute(0, 2, 3, 1))

        loc = torch.cat([layer.reshape(layer.shape[0], -1) for layer in loc], 1)
        conf = torch.cat([layer.reshape(layer.shape[0], -1) for layer in conf], 1)

        if training is True:
            output = (
                loc.reshape(loc.shape[0], -1, 4),
                conf.reshape(conf.shape[0], -1, self.num_class)
            )
        else:
            output = (
                loc.reshape(loc.shape[0], -1, 4),
                F.softmax(conf.reshape(conf.shape[0], -1, self.num_class), dim = 2)
            )
        return output

def _conv_dw(inp, oup, stride=1, padding=0, expand_ratio=1):
    """
    Introduction
    ------------
        采用MobileNet的结构减少参数，压缩模型
    Parameters
    ----------
        inp: 输入通道数
        oup: 输出通道数
        stride: 卷积步长
        padding: 卷积padding大小
        expand_ratio: 维度扩充比例
    Returns:
    -------
        压缩模型参数的结构
    """
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(),
        # dw
        nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups = oup * expand_ratio, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(),
        # pw-linear
        nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


def add_extras(base, feature_layer, mbox, num_classes):
    """
    Introduction
    ------------
        将基础模型特征基础上提取多尺度特征，再使用卷积进行box坐标和类别的回归
    Parameters
    ----------
        base: 基础模型
        feature_layer: 特征层参数
        mbox: 每层特征预测box的数量
        num_classes: 数据集类别数量
    """
    extra_layers = []
    loc_layers = []
    conf_layers = []
    input_channels = None
    for layer, depth, box_num in zip(feature_layer[0], feature_layer[1], mbox):
        if layer == 'S':
            extra_layers.append(_conv_dw(input_channels, depth, stride = 2, padding = 1, expand_ratio = 1))
            input_channels = depth
        elif layer == '':
            extra_layers.append(_conv_dw(input_channels, depth, stride = 1, expand_ratio = 1))
            input_channels = depth
        else:
            input_channels = depth
        loc_layers.append(nn.Conv2d(input_channels, box_num * 4, kernel_size = 3, padding = 1))
        conf_layers.append(nn.Conv2d(input_channels, box_num * num_classes, kernel_size=3, padding = 1))
    return base, extra_layers, (loc_layers, conf_layers)


def build_ssd_lite(base, feature_layer, mbox, num_classes):
    """
    Introduction
    ------------
        构建ssd Lite模型结构
    Parameters
        base: 使用的基础网络
        feature_layer: 特征层的参数
        mbox: 每个feature map的像素点对应的mbox的数量
        num_classes: 样本类别数量
    Returns
    -------
        SSD Lite模型结构
    """
    base_, extras_, head_ = add_extras(base, feature_layer, mbox, num_classes)
    return SSDLite(base_, extras_, head_, feature_layer, num_classes)

