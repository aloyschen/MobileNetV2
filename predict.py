import os
import cv2
import torch
from utils import config
import matplotlib.pyplot as plt
from models.SSDLite import build_ssd_lite
from models.MobileNet import MobileNetV2
from models.Modules import Detect, PriorBox
import torchvision.transforms as transforms

def predict(image_file):
    """
    Introduction
    ------------
        加载模型进行预测
    Parameters
    ----------
        image_file: 图片路径
    """
    checkpoint = torch.load('./checkpoint.pth.tar', map_location='cpu')
    base_model = MobileNetV2(width_multiplier=config.width_multiplier)
    model = build_ssd_lite(base_model, config.feature_layer['MobileNetV2'], config.mbox['MobileNetV2'], num_classes=2)
    prior_box = PriorBox(config.Prior_box['MobileNetV2'])
    priors = prior_box.forward()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    detector = Detect(config.num_classes, top_k=config.top_k, conf_thresh=config.conf_thresh, nms_thresh=config.nms_thresh, variances=config.Prior_box['MobileNetV2']['variance'])
    for file in image_file:
        image = cv2.imread(file)
        image = cv2.resize(image, (300, 300))
        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0)
        start = time.time()
        output = model(image_tensor, training = False)
        _, conf = output
        detections = detector.forward(output, priors)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for detect_image in detections:
            for detect_class in detect_image[1:]:
                for object in detect_class:
                    score = object[0]
                    if score > config.conf_thresh:
                        print(score)
                        box = object[1:]
                        cv2.rectangle(image, (box[0] * 300, box[1] * 300), (box[2] *300, box[3] * 300), [0, 255, 0], 2)
        print('detect time: {}'.format(time.time() - start))
        plt.imshow(image)
        plt.show()
    else:
        print('Image path is wrong')


if __name__ == '__main__':
    import time
    start = time.time()
    files = ['./5.jpg', './6.jpg', './7.jpg', './8.jpg']
    predict(files)
    print(time.time() - start)