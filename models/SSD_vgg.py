from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Modules import L2Norm


class SSD(nn.Module):
    def __init__(self, BaseModel, extras, head, num_class, size):
        """
        Introduction
        ------------
            Single Shot Multibox Architecture
            SSD模型初始化函数，在base model提取特征基础上，增加multi box卷积获取类别概率，box坐标等
        Parameters
        ----------
            BaseModel: vgg基础网络结构
            extras: ssd添加的多尺度特征提取层
            head: multibox中的loc和class conf layer
            num_class: 训练集类别数量
            size: 图片的大小，包括300，500尺寸
        """
        super(SSD, self).__init__()
        self.num_class = num_class
        self.size = size
        self.base = nn.ModuleList(BaseModel)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax()


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
        # vgg基础模型
        for index in range(23):
            x = self.base[index](x)
        # 对conv4_3进行L2norm归一化处理，conv4_3 shape [38, 38, 512]
        sources.append(self.L2Norm(x))
        # 获取fc7特征层，fc7 shape [19, 19, 1024]
        for index in range(23, len(self.base)):
            x  = self.base[index](x)
        sources.append(x)
        # 添加多尺度特征层[10, 10, 512], [5, 5, 256], [3, 3, 256], [1, 1, 256]
        for index, value in enumerate(self.extras):
            x = F.relu(value(x))
            if index % 2 == 1 :
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
            print(loc.shape)
        else:
            output = (
                loc.reshape(loc.shape[0], -1, 4),
                self.softmax(conf.reshape(conf.shape[0], -1, self.num_class))
            )
        return output



def build_vgg(cfg, input_channels):
    """
    Introduction
    ------------
        建立基础模型VGG的网络结构
    Parameters
    ----------
        cfg: 模型结构配置列表
        input_channels: 输入图像的通道数
    Returns
    -------
        net: 模型结构列表
    """
    net = []
    for value in cfg:
        if value == 'M':
            net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # 当池化层降采样无法整除时，需要将cell_mode置为True， 保留边缘的像素值
        elif value == 'C':
            net.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            net.append(nn.Conv2d(input_channels, value, kernel_size=3, padding=1))
            net.append(nn.ReLU())
            input_channels = value
    net.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
    # 使用空洞卷积
    net.append(nn.Conv2d(512, 1024, kernel_size = 3, padding = 6, dilation = 6))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(1024, 1024, kernel_size = 1))
    net.append(nn.ReLU())
    return net


def multiScale_extra(cfg, input_channels, size = 300):
    """
    Introduction
    ------------
        在vgg模型基础上添加多尺度特征提取层
    """
    net = []
    flag = False
    input_channels = input_channels
    for index, value in enumerate(cfg):
        if input_channels != 'S':
            if value == 'S':
                net.append(nn.Conv2d(input_channels, cfg[index + 1], kernel_size = (1, 3)[flag], stride = 2, padding = 1))
            else:
                net.append(nn.Conv2d(input_channels, value, kernel_size = (1, 3)[flag]))
            flag = not flag
        input_channels = value
    if size == 512:
        net.append(nn.Conv2d(input_channels, 128, kernel_size = 1, stride = 1))
        net.append(nn.Conv2d(128, 256, kernel_size = 4, stride = 1, padding = 1))
    return net


def multi_box(vgg, extras, cfg, num_classes):
    """
    Introduction
    ------------
        根据特征层提取box坐标和类别
    Parameters
    ----------
        vgg: 基础vgg模型
        extras: 多尺度特征层
        cfg: 配置参数
        num_classes: 数据集类别数量
    """
    loc_layers = []
    class_layers = []
    vgg_source = [21, -2]
    for index, value in enumerate(vgg_source):
        loc_layers.append(nn.Conv2d(vgg[value].out_channels, cfg[index] * 4, kernel_size = 3, padding = 1))
        class_layers.append(nn.Conv2d(vgg[value].out_channels, cfg[index] * num_classes, kernel_size = 3, padding = 1))
    for index, value in enumerate(extras[1::2], 2):
        loc_layers.append(nn.Conv2d(value.out_channels, cfg[index] * 4, kernel_size = 3, padding = 1))
        class_layers.append(nn.Conv2d(value.out_channels, cfg[index] * num_classes, kernel_size = 3, padding = 1))
    return vgg, extras, (loc_layers, class_layers)

def build_model(size = 300, num_classes = 21):
    """
    Introduction
    ------------
        构建ssd模型
    Parameters
    ----------
        size: 图像大小
        num_classes: 数据集类别数量
    Returns
    -------
        SSD模型
    """
    if size != 300 and size != 500:
        print("Error: Only ssd300 and ssd512 is supported correctly")
        return
    else:
        vgg, extra, head = multi_box(build_vgg(config.vgg_base[str(size)], 3),
                                     multiScale_extra(config.extras[str(size)], 1024, size = size),
                                     config.mbox[str(size)], num_classes = num_classes)
        return SSD(vgg, extra, head, num_class = num_classes, size = size)



