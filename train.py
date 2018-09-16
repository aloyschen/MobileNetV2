import os
import config
import torch
from model import MobileNetV2
from cifar100data import CIFAR100data
import torch.nn as nn
from torch.optim import Adam
from torch.optim.rmsprop import RMSprop
from tqdm import tqdm
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils import AverageTracker, compute_accuracy
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    dataset = CIFAR100data()
    train_loader, val_loader = dataset.train_dataloader, dataset.val_dataloader
    loss = nn.CrossEntropyLoss().cuda()
    #optimizer = Adam(model.parameters(), config.learning_rate)
    optimizer = RMSprop(model.parameters(), config.learning_rate, momentum = config.momentum, weight_decay = config.weight_decay)
    summary_writer = SummaryWriter(log_dir = config.log_dir)
    best_top1 = 0.
    for epoch in range(config.Epoch_num):
        tqdm_batch = tqdm(train_loader, desc = "Epoch-" + str(epoch) + "-")
        learning_rate = config.learning_rate * (config.learning_rate_decay ** epoch)
        for params_group in optimizer.param_groups:
            params_group['lr'] = learning_rate
        loss_value, top1, top5 = AverageTracker(), AverageTracker(), AverageTracker()
        # 使batchnormlization生效
        model.train()
        for data, target in tqdm_batch:
            data, target = data.cuda(), target.cuda()
            data_var, target_var = Variable(data), Variable(target)
            output = model(data_var)
            cur_loss = loss(output, target_var)
            # 将梯度置为0，为下次计算做准备
            optimizer.zero_grad()
            # 反向传播计算梯度
            cur_loss.backward()
            # 更新参数
            optimizer.step()
            # 计算准确率
            cur_acc1, cur_acc5 = compute_accuracy(output.data, target_var, topk = (1, 5))
            loss_value.update(cur_loss.data.cpu().item())
            top1.update(cur_acc1[0].item())
            top5.update(cur_acc5[0].item())
        summary_writer.add_scalar("epoch-loss", loss_value.avg, epoch)
        summary_writer.add_scalar("epoch-top1-accuracy", top1.avg, epoch)
        summary_writer.add_scalar("epoch-top5-accuracy", top5.avg, epoch)

        tqdm_batch.close()
        print("Epoch-" + str(epoch) + " | " + "loss: " + str(loss_value.avg) + " - acc-top1: " + str(top1.avg)[:7] + "- acc-top5: " + str(top5.avg)[:7])

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
    loss_value, top1, top5 = AverageTracker(), AverageTracker(), AverageTracker()
    # 使batchnormlization生效
    model.train()
    for data, target in dataloader:
        data_var, target_var = Variable(data.cuda()), Variable(target.cuda())
        output = model(data_var)
        cur_loss = loss(output, target_var)

        # 计算准确率
        cur_acc1, cur_acc5 = compute_accuracy(output.data, target_var, topk=(1, 5))
        loss_value.update(cur_loss.data.cpu().item())
        top1.update(cur_acc1[0].item())
        top5.update(cur_acc5[0].item())
    summary_writer.add_scalar("test-loss", loss_value.avg, epoch)
    summary_writer.add_scalar("test-top1-accuracy", top1.avg, epoch)
    summary_writer.add_scalar("test-top5-accuracy", top5.avg, epoch)

    print("test Results " + str(epoch) + " | " + "loss: " + str(loss_value.avg) + " - acc-top1: " + str(top1.avg)[:7] + "- acc-top5: " + str(top5.avg)[:7])


if __name__ == '__main__':
    train()
