import torch
import numpy as np

class AverageTracker:
    """
    Introduction
    ------------
        求平均值
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_accuracy(output, target, topk=(1,)):
    """
    Introduction
    ------------
        计算模型TopK的准确率
    Parameters
    ----------
        output: 模型输出
        target: 真是标签
        topk: K值
    Returns
    -------
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, idx = output.topk(maxk, 1, True, True)
        idx = idx.t()
        correct = idx.eq(target.view(1, -1).expand_as(idx))

        acc_arr = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc_arr.append(correct_k.mul_(100.0 / batch_size))
    return acc_arr
