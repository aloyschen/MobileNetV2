import os
import torch
import time
import sys
from utils import config
from models.MobileNet import MobileNetV2
from dataset.ImageNetData import DataReader
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils.train_utils import AverageTracker, compute_accuracy
from tensorboardX import SummaryWriter

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index
sys.path.append("..")

def train():
    """
    Introduction
    ------------
        训练MobileNet模型
    """

    model = MobileNetV2(config.num_classes)
    model.cuda()
    cudnn.enabled = True
    cudnn.benchmark = True
    dataset = DataReader()
    train_loader, val_loader = dataset.train_dataloader, dataset.val_dataloader
    loss = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), config.learning_rate, momentum = config.momentum, weight_decay = config.weight_decay)
    summary_writer = SummaryWriter(log_dir = config.log_dir)
    best_top1 = 0.
    for epoch in range(config.Epoch_num):
        batch_time, losses, top1, top5 = AverageTracker(), AverageTracker(), AverageTracker(), AverageTracker()
        adjust_learning_rate(optimizer, epoch)
        # 使batchnormlization生效
        model.train()
        end = time.time()
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            cur_loss = loss(output, target)
            # 将梯度置为0，为下次计算做准备
            optimizer.zero_grad()
            # 反向传播计算梯度
            cur_loss.backward()
            # 更新参数
            optimizer.step()
            # 计算准确率
            cur_acc1, cur_acc5 = compute_accuracy(output, target, topk = (1, 5))
            losses.update(cur_loss.item(), data.size(0))
            top1.update(cur_acc1[0], data.size(0))
            top5.update(cur_acc5[0], data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % config.print_freq == 0:
                print('Epoch: {}/{} Batch: {}/{} Loss: {:.4f} {:.4f} Prec@1: {:.2f} {:.2f} Prec@5: {:.2f} {:.2f} Time: {:.4f} {:.4f}'.format(epoch, config.Epoch_num, idx, len(train_loader), losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg, batch_time.val, batch_time.avg))

        summary_writer.add_scalar("epoch-loss", losses.avg, epoch)
        summary_writer.add_scalar("epoch-top1-accuracy", top1.avg, epoch)
        summary_writer.add_scalar("epoch-top5-accuracy", top5.avg, epoch)

        eval(model = model, loss = loss, dataloader = val_loader, summary_writer = summary_writer, epoch = epoch)

        if(top1.avg > best_top1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_top1': best_top1,
                'optimizer': optimizer.state_dict(),
            }, config.model_dir + 'checkpoint.pth.tar')


def eval(model, loss, dataloader, summary_writer, epoch):
    """
    Introduction
        使用验证集对模型进行评估
    Parameters
    ----------
        dataloader: 数据读取
        epoch: 迭代次数
    """
    batch_time, losses, top1, top5 = AverageTracker(), AverageTracker(), AverageTracker(), AverageTracker()
    # 使batchnormlization生效
    model.eval()
    end = time.time()
    for idx, (data, target) in enumerate(dataloader):
        data_var, target_var = Variable(data.cuda()), Variable(target.cuda())
        output = model(data_var)
        cur_loss = loss(output, target_var)

        # 计算准确率
        cur_acc1, cur_acc5 = compute_accuracy(output.data, target_var, topk=(1, 5))
        losses.update(cur_loss.item(), data_var.size(0))
        top1.update(cur_acc1[0], data_var.size(0))
        top5.update(cur_acc5[0], data_var.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.print_freq == 0:
            print('Epoch: {}/{} Batch: {}/{} Loss: {:.4f} {:.4f} Prec@1: {:.2f} {:.2f} Prec@5: {:.2f} {:.2f} Time: {:.2f} {:.2f}'.format(epoch, config.Epoch_num, idx, len(dataloader), losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg, batch_time.val, batch_time.avg))
    summary_writer.add_scalar("test-loss", losses.avg, epoch)
    summary_writer.add_scalar("test-top1-accuracy", top1.avg, epoch)
    summary_writer.add_scalar("test-top5-accuracy", top5.avg, epoch)


def adjust_learning_rate(optimizer, epoch):
    """
    Introduction
    ------------
        每10个epoch调整一次学习率
    Parameters
    ----------
        optimizer:优化器
        epoch:当前epoch
    """
    lr = config.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
