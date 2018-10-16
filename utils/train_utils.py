import torch
from utils import config
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import torch.nn.init as init

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

def weights_init(m):
    """
    Introduction
    ------------
        初始化权重矩阵
    Parameters
    ----------
        m: 需要初始化的模型参数
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

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
    lr = config.learning_rate * (0.8 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def summary(model, input_size, batch_size=-1, device="cuda"):
    """
    Introduction
    ------------
        获取模型每层的输出参数和具体的训练参数个数
    Parameters
    ----------
        model: 模型结构
        input_size: 输入尺寸 shape为【channels, weight, height】
        batch_size: batch的大小
        device: 是否使用cuda
    """

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")

def box_iou(detect, ground_truth):
    """
    Introduction
    ------------
        计算每个预测的结果和一张图片中每个ground_truth的iou
    Parameters
    ----------
        detect: 预测的box结果 [xmin, ymin, xmax, ymax]
        ground_truth: 每个图片中所有ground truth box [xmin, ymin, xmax, ymax]
    Returns

    """
    det_size = (detect[2] - detect[0]) * (detect[3] - detect[1])
    detect = detect.view(1, 4)
    iou = []
    ground_truth = torch.Tensor(ground_truth)
    for gt in ground_truth:
        gt_size = (gt[2] - gt[0]) * (gt[3] - gt[1])
        inter_max = torch.max(detect, gt)
        inter_min = torch.min(detect, gt)
        inter_size = max(inter_min[0][2] - inter_max[0][0], 0.) * max(inter_min[0][3] - inter_max[0][1], 0.)

        _iou = inter_size / (det_size + gt_size - inter_size)
        iou.append(_iou)

    return iou


def compute_fp_tp(pred, ground_truth, npos, label, score, iou_threshold = 0.5):
    """
    Introduction
    ------------
        计算模型的MAP
    Parameters
    ----------
        pred: 模型预测经过nms结果 [batch, num_class, top_k, 5]
        ground_truth: 真实值 [batch, box_num, 5]
        iou_threshold: iou阈值判断预测正确
    """
    for detect_per_image, gt_per_image in zip(pred, ground_truth):
        for class_index, detect_class in enumerate(detect_per_image):
            iou_c = []
            score_c = []
            gt_class_boxes = [object[:4] for object in gt_per_image if int(object[4]) == class_index]
            for detect_box in detect_class:
                if detect_box[0] > config.conf_thresh:
                    score_c.append(detect_box[0])
                    # 背景label无box
                    if len(gt_class_boxes) > 0:
                        _iou = box_iou(detect_box[1:], gt_class_boxes)
                        iou_c.append(_iou)

            # 没有检测到物体
            if len(iou_c) == 0:
                npos[class_index] += len(gt_class_boxes)
                continue

            labels_c = [0] * len(score_c)
            if len(gt_class_boxes) > 0:
                max_overlap_gt_ids = np.argmax(np.array(iou_c), axis = 1)
                is_gt_box_detected = np.zeros(len(gt_class_boxes), dtype = bool)
                for iters in range(len(labels_c)):
                    gt_id = max_overlap_gt_ids[iters]
                    if iou_c[iters][gt_id] >= iou_threshold:
                        if not is_gt_box_detected[gt_id]:
                            labels_c[iters] = 1
                            is_gt_box_detected[gt_id] = True
            npos[class_index] += len(gt_class_boxes)
            label[class_index] += labels_c
            score[class_index] += score_c
    return label, score, npos

def compute_mAP(_label, _score, _npos):
    """
    Introduction
    ------------
        计算预测结果所有类别的mAP和
    Parameters
    ----------
        _label: 每个类别中预测正确和错误的标签 [num_class, detect_box_num]
        _score: 每个box属于一个类别的概率值 [num_class, detect_box_num]
        _npos: 所有该类别的box总数 [num_class, 1]
    Returns
    -------
        mAP: 所有类别mAP和
    """
    recall = []
    precision = []
    ap = []
    for labels, scores, npos in zip(_label[1:], _score[1:], _npos[1:]):
        sorted_indices = np.argsort(scores)
        sorted_indices = sorted_indices[::-1]
        labels = np.array(labels).astype(int)
        true_positive_labels = labels[sorted_indices]
        false_positive_labels = 1 - true_positive_labels
        tp = np.cumsum(true_positive_labels)
        fp = np.cumsum(false_positive_labels)

        rec = tp.astype(float) / float(npos)
        prec = tp.astype(float) / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap += [compute_average_precision(prec, rec)]
        recall += [rec]
        precision += [prec]
    mAP = np.nanmean(ap)
    return mAP

def compute_average_precision(precision, recall):
  """Compute Average Precision according to the definition in VOCdevkit.
  Precision is modified to ensure that it does not decrease as recall
  decrease.
  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls
  Raises:
    ValueError: if the input is not of the correct format
  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.
  """
  if precision is None:
    if recall is not None:
      raise ValueError("If precision is None, recall must also be None")
    return np.NAN

  if not isinstance(precision, np.ndarray) or not isinstance(recall,
                                                             np.ndarray):
    raise ValueError("precision and recall must be numpy array")
  if precision.dtype != np.float or recall.dtype != np.float:
    raise ValueError("input must be float numpy array.")
  if len(precision) != len(recall):
    raise ValueError("precision and recall must be of the same size.")
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("Precision must be in the range of [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range of [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Preprocess precision to be a non-decreasing array
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
    This part makes the precision monotonically decreasing
    (goes from the end to the beginning)
    matlab:  for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
    This part creates a list of indexes where the recall changes
    matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
    The Average Precision (AP) is the area under the curve
    (numerical integration)
    matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def my_collate_fn(batch):
    """
    Introduction
    ------------
        对dataset解析
    parameters
    ----------
        batch: 每个batch的数据
    """
    images = torch.stack(list(map(lambda x: torch.tensor(x[0]), batch)))
    coordinates = list(map(lambda x: x[1], batch))
    return images, coordinates