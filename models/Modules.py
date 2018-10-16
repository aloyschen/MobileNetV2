import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from utils.box_utils import match_priors, decode, nms
from itertools import product as product

class L2Norm(nn.Module):
    def __init__(self, input_channels, scale):
        super(L2Norm, self).__init__()
        self.input_channels = input_channels
        self.gamma = scale or None
        self.eps = 1e-8
        self.weight = nn.Parameter(torch.Tensor(self.input_channels))
        nn.init.constant_(self.weight, self.gamma)


    def forward(self, input):
        norm = input.pow(2).sum(dim = 1, keepdim = True).sqrt() + self.eps
        input = input / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(input) * input
        return out


class PriorBox(object):
    """
    Introduction
    ------------
        根据feature map生成box坐标
    """
    def __init__(self,cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['image_size']
        # 每个feature map 上像素点对应的prior box的数量
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or 0.1
        self.featrue_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for value in self.variance:
            if value <= 0:
                raise ValueError("Variances must be greater than 0")

    def forward(self):
        """
        Introduction
        ------------
            计算prior box坐标
        """
        mean = []
        for index, value in enumerate(self.featrue_maps):
            for i, j in product(range(value), repeat = 2):
                f_k = self.image_size / self.steps[index]
                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k
                s_k = self.min_sizes[index] / self.image_size
                mean.append([cx, cy, s_k, s_k])
                if self.max_sizes:
                    s_k_prime = sqrt(s_k * (self.max_sizes[index] / self.image_size))
                    mean.append([cx, cy, s_k_prime, s_k_prime])
                for ar in self.aspect_ratios[index]:
                    mean.append([cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)])
                    mean.append([cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)])
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, neg_pos_ratio):
        """
        Introduction
        ------------
            计算多尺度loss
        Parameters
        ----------
            num_classes: 数据集类别数量
        """
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.variances = [0.1, 0.2]
        self.neg_pos_ratio = neg_pos_ratio


    def forward(self, predictions, priors, targets):
        """
        Parameters
        ----------
            predictions: 包含位置预测值和类别预测值
            priors: 每个feature map上生成的prior box
            targets: 每张图片上对应的box坐标和类别id，最后一维是类别坐标
        """
        loc_data, conf_data = predictions
        num = loc_data.shape[0]
        num_priors = priors.shape[0]
        priors = priors
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors).cuda()
        for idx in range(num):
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]
            truths = torch.Tensor(truths)
            labels = torch.Tensor(labels)
            defaults = priors.detach()
            match_priors(truths, defaults, labels, self.threshold, self.variances, loc_t, conf_t, idx)
        pos = conf_t > 0
        # 广播扩充维度
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_pred = loc_data[pos_idx].view(-1, 4)
        loc_truth = loc_t[pos_idx].view(-1, 4)
        # 计算正样本的location位置损失
        loss_location = F.smooth_l1_loss(loc_pred, loc_truth, reduction = 'sum')
        # 计算所有样本的类别损失
        batch_conf = conf_data.reshape([-1, self.num_classes])
        loss_conf = F.cross_entropy(batch_conf, conf_t.reshape([-1]), ignore_index = -1, reduction = 'none')
        loss_conf = loss_conf.reshape([num, -1])
        # 将负样本对应的loss_conf排序
        pos_loss_conf = loss_conf[pos]
        loss_conf[pos] = 0
        _, loss_idx = loss_conf.sort(1, descending = True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim = True)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max = pos.shape[1] - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg_loss_conf = loss_conf[neg]
        loss_conf = pos_loss_conf.sum() + neg_loss_conf.sum()
        N = num_pos.data.sum().float()
        loss_conf = loss_conf / N
        loss_location = loss_location / N
        return loss_location, loss_conf


class Detect(nn.Module):
    def __init__(self, num_classes, top_k, variances, conf_thresh, nms_thresh):
        """
        Introduction
        ------------
            模型的预测模块
        Parameters
        ----------
            num_classes: 数据集类别数量
            bkg_label: 非物体标签
            top_k: 每种类别保留多少个box
            variances: decode 预测坐标时的偏差
            conf_thresh: 物体概率的阈值
            nms_thresh: nms阈值
        """
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.variances = variances
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k


    def forward(self, predictions, priors):
        """
        Introduction
        ------------
            分别对每种类别的Box做nms, 选择每种类别的保留topk box
        Parameters
        ----------
            loc_data: 预测的box坐标
            conf_data: 预测的物体概率值
            prior_data: prior box坐标
        Returns
        -------
            output: 每种类别的topk box坐标和类别
        """
        loc, conf = predictions
        loc_data = loc.detach()
        conf_data = conf.detach()
        num = loc_data.shape[0]
        prior_data = priors.detach()
        num_priors = prior_data.shape[0]
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.reshape([num, num_priors, self.num_classes]).transpose(2, 1)
        for index in range(num):
            decoded_boxes = decode(loc_data[index], prior_data, self.variances)
            conf_scores = conf_preds[index].clone()
            # 每个类别做nms
            for class_index in range(1, self.num_classes):
                class_mask = conf_scores[class_index].gt(self.conf_thresh)
                scores = conf_scores[class_index][class_mask]
                if scores.shape[0] == 0:
                    continue
                loc_mask = class_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[loc_mask].reshape([-1, 4])
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[index, class_index, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        return output
