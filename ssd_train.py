import os
import time
from utils import config
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
from utils.train_utils import *
import torch.backends.cudnn as cudnn
from models.Modules import PriorBox, MultiBoxLoss, Detect
from models.MobileNet import MobileNetV2
from models.SSDLite import build_ssd_lite
import torchvision.transforms as transforms
from dataset.FDDBData import FDDBDataset

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

torch.set_default_tensor_type(torch.cuda.FloatTensor)

def train():
    """
    Introduction
    ------------
        训练模型
    """
    best_mAP = 0
    train_dataset = FDDBDataset(config.fddb_train_dir, config.fddb_annotation_file, training = True, transform = [transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size = config.train_batch, shuffle = True, num_workers = 2, pin_memory = True, collate_fn = my_collate_fn)
    print('=====Building Model====')
    base_model = MobileNetV2(width_multiplier = config.width_multiplier)
    model = build_ssd_lite(base_model, config.feature_layer['MobileNetV2'], config.mbox['MobileNetV2'], num_classes = 2)
    prior_box = PriorBox(config.Prior_box['MobileNetV2'])
    priors = prior_box.forward()
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)
    criterion = MultiBoxLoss(num_classes = config.num_classes, overlap_thresh = config.overlap_thresh, neg_pos_ratio = config.neg_pos_ratio)
    summary_writer = SummaryWriter(log_dir = config.log_dir)
    if config.use_gpu:
        model.cuda()
        priors.cuda()
        cudnn.benchmark = True
        cudnn.enabled = True
    if not config.resume_net:
        print("Initializing Weights")
        model.extras.apply(weights_init)
        model.loc.apply(weights_init)
        model.conf.apply(weights_init)
    else:
        if not os.path.exists(config.checkpoint_dir):
            print("no checkpoint file")
        else:
            print("=> loading checkpoint '{}'".format(config.checkpoint_dir))
            checkpoint = torch.load(config.checkpoint_dir)
            config.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(config.checkpoint_dir, checkpoint['epoch']))
    for epoch in range(config.Epoch_num):
        batch_time, loc_losses, conf_losses, top1, top5 = AverageTracker(), AverageTracker(), AverageTracker(), AverageTracker(), AverageTracker()
        adjust_learning_rate(optimizer, epoch)
        # 使batch norm生效
        model.train()
        end = time.time()
        for idx, (image, target) in enumerate(train_loader):
            image = image.cuda()
            output = model(image, training = True)
            loss_l, loss_c = criterion(output, priors, target)
            cur_loss = loss_l + loss_c
            # 将梯度置为0，为下次计算做准备
            optimizer.zero_grad()
            # 反向传播计算梯度
            cur_loss.backward()
            # 更新参数
            optimizer.step()
            loc_losses.update(loss_l.item(), image.size(0))
            conf_losses.update(loss_c.item(), image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % config.print_freq == 0:
                print('Epoch: {}/{} Batch: {}/{} loc Loss: {:.4f} {:.4f} conf loss: {:.2f} {:.2f} Time: {:.4f} {:.4f}'.format(epoch, config.Epoch_num, idx, len(train_loader), loc_losses.val, loc_losses.avg, conf_losses.val, conf_losses.avg, batch_time.val, batch_time.avg))

        summary_writer.add_scalar("epoch-loc-loss", loc_losses.avg, epoch)
        summary_writer.add_scalar("epoch-loc-loss", conf_losses.avg, epoch)
        mAP = eval(model, priors)
        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        if is_best:
            torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mAP': best_mAP,
            'optimizer': optimizer.state_dict(),
        }, config.checkpoint_dir + 'checkpoint.pth.tar')


def eval(model, priors):
    """
    Introduction
    ------------
        评估模型, 计算模型预测结果的MAP
    """
    model.eval()
    detector = Detect(config.num_classes, top_k = config.top_k, conf_thresh = config.conf_thresh, nms_thresh = config.nms_thresh, variances = config.Prior_box['MobileNetV2']['variance'])
    eval_dataset = FDDBDataset(config.fddb_train_dir, config.fddb_annotation_file, training = False, transform = [transforms.ToTensor()])
    val_dataloader = DataLoader(eval_dataset, batch_size = config.val_batch, shuffle = False, num_workers = 2, pin_memory = True, collate_fn = my_collate_fn)
    label = [list() for _ in range(config.num_classes)]
    score = [list() for _ in range(config.num_classes)]
    npos = [0] * config.num_classes
    for index, (image, targets) in enumerate(val_dataloader):
        image = image.cuda()
        output = model(image, training = False)
        detections = detector.forward(output, priors)
        label, score, npos = compute_fp_tp(detections, targets, npos, label, score)
    mAP = compute_mAP(label, score, npos)
    print('Eval mAP: {}'.format(mAP))
    return mAP

if __name__ == '__main__':
    train()


