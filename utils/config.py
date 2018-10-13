gpu_index = "1"
Epoch_num = 60
learning_rate = 0.01
print_freq = 3
train_num = 12880
val_num = 3226
image_height = 300
train_batch = 18
val_batch = 18
num_classes = 2
weight_decay = 5e-4
momentum = 0.9
overlap_thresh = 0.5
conf_thresh = 0.3
nms_thresh = 0.5
top_k = 100
width_multiplier = 1
neg_pos_ratio = 3
use_gpu = True
resume_net = False
log_dir = './logs'
checkpoint_dir = './model_ckpt/'
train_dir = '/data0/dataset/ImageNet/TrainDataset'
val_dir = '/data0/dataset/ImageNet/ValDataset'
fddb_train_dir = '../data/FDDB-images/'
fddb_annotation_file = '/data0/gaochen3/FaceDetect/data/FDDB-folds/FDDB_annotations.txt'
wider_face_train_dir = '../data/WIDER_train/images/'
wider_face_val_dir = '../data/WIDER_val/images/'
wider_face_train_annotations = './data/wider_face_split/wider_face_train_bbx_gt.txt'
wider_face_val_annotations = './data/wider_face_split/wider_face_val_bbx_gt.txt'
vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
feature_layer = {
    'MobileNetV2': [[13, 17, 'S', 'S', 'S', 'S'], [96, 320, 512, 256, 256, 128]]
}
mbox = {
    '300': [6, 6, 6, 6, 4, 4],
    '512': [6, 6, 6, 6, 6, 4, 4],
    'MobileNetV2' : [8, 8, 8, 8, 6, 6]
}
Prior_box = {
    'MobileNetV2' : {
        'image_size' : 300,
        'steps' : [16, 32, 64, 100, 150, 300],
        'min_sizes' : [45, 90, 135, 180, 225, 270],
        'max_sizes' : [90, 135, 180, 225, 270, 315],
        'feature_maps' : [19, 10, 5, 3, 2, 1],
        'aspect_ratios' :  [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]],
        'variance': [0.1, 0.2],
        'clip' : True
    }
}