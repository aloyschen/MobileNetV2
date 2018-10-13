import torch

def box_iou(box1, box2):
    """
    Introduction
    ------------
        计算两个box的iou
    Parameters
    ----------
        box1: box的坐标[xmin, ymin, xmax, ymax], shape[box1_num, 4]
        box2: box的坐标[xmin, ymin, xmax, ymax], shape[box2_num, 4]
    Returns
    -------
        iou: 两个box的iou数值, shape[box1_num, box2_num]
    """
    box_num1 = box1.shape[0]
    box_num2 = box2.shape[0]
    max_xy = torch.min(box1[:, 2:].unsqueeze(1).expand(box_num1, box_num2, 2), box2[:, 2:].unsqueeze(0).expand(box_num1, box_num2, 2))
    min_xy = torch.max(box1[:, :2].unsqueeze(1).expand(box_num1, box_num2, 2), box2[:, :2].unsqueeze(0).expand(box_num1,box_num2, 2))
    inter_xy = torch.clamp((max_xy - min_xy), min = 0)
    inter_area = inter_xy[:, :, 0] * inter_xy[:, :, 1]
    box1_area = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])).unsqueeze(1).expand_as(inter_area)
    box2_area = ((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])).unsqueeze(0).expand_as(inter_area)
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def match_priors(truths, priors, labels, threshold, variances, loc_t, conf_t, idx):
    """
    Introduction
    ------------
        匹配priors box和truth box, 计算每个prior box和truth box重合iou，如果最大iou大于阈值则认为是正样本
    Parameters
    ----------
        truths: ground truth box坐标
        priors: 预选框坐标
        labels: ground truth box对应的类别
        threshold: iou阈值
        variances: 坐标变换修正
        loc_t: 正样本坐标
        conf_t: 每个prior box的类别，0为不含物体的背景
        idx: 每张训练图片的index
    """
    # 将priors坐标转换为[xmin, ymin, xmax, ymax]
    priors_ = torch.cat((priors[:, :2] - priors[:, 2:] / 2, priors[:, :2] + priors[:, 2:] / 2), 1)
    iou = box_iou(truths, priors_)
    # 找到每个truth box对应的iou最大的prior box
    best_priors_iou, best_priors_idx = iou.max(1, keepdim = True)
    # 找到每个prior box对应的iou最大的truth box
    best_truth_iou, best_truth_idx = iou.max(0, keepdim = True)
    best_truth_iou.squeeze_(0)
    best_truth_idx.squeeze_(0)
    best_priors_idx.squeeze_(1)
    best_priors_iou.squeeze_(1)
    # 将和truth box重合度最高的prior box标记出来
    best_truth_iou.index_fill(0, best_priors_idx, 2)
    # 确保每个prior box只对应一个truth box
    for j in range(best_priors_idx.shape[0]):
        best_truth_idx[best_priors_idx[j]] = j
    # 获取每个prior box对应的truth box
    matches = truths[best_truth_idx]
    matches[best_truth_iou == 0] = 0
    conf = labels[best_truth_idx]
    conf[best_truth_iou < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf

def encode(matches, priors, variances):
    """
    Introduction
    ------------
        对正样本的prior box坐标进行encode变换，使用每个prior box和truth box的偏移做为训练数据集的坐标
    Parameters
    ----------
        matches: 每个prior box匹配的truth box坐标
        priors: 预选框坐标
        variances: 个人理解是为了控制box坐标范围
    Returns
    -------
        encoded boxes: 转换后的坐标
    """
    box_xy = (matches[:, :2] + matches[:, 2:]) / 2 - priors[:, :2]
    box_xy /= (variances[0] * priors[:, 2:])
    box_wh = (matches[:, 2:] - matches[:, :2]) / priors[:, 2:]
    box_wh = torch.log(box_wh) / variances[1]
    return torch.cat([box_xy, box_wh], 1)

def decode(loc, priors, variances):
    """
    Introduction
    ------------
        对模型预测出的坐标转换回来，对应encode的逆变换
    Parameters
    ----------
        loc: 预测box坐标值
        priors: 预选框坐标值
        variances: 坐标修正系数
    Returns
    -------
        boxes: 变换后的坐标
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors, 4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
