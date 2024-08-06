"""
Teacher free KD, main.py
"""
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils1
import model.resnet as resnet
import model.mobilenetv2 as mobilenet

import torchvision.models as models
from my_loss_function import loss_DIST
from train_kd_eval_openset import train_and_evaluate, train_and_evaluate_kd

#  modified
from data.open_set_datasets import get_class_splits, get_datasets
from torch.utils.data import DataLoader
from utils.utils import str2bool


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_exp/resnet18/', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'last'
parser.add_argument('--num_class', default=100, type=int, help="number of classes")
parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
parser.add_argument('--dataset', type=str, default='tinyimagenet', help="The training dataset.")
parser.add_argument('--transform', type=str, default='rand-augment', help="default, rand-augment, cutout, randomized_quantization, augmix")
parser.add_argument('--split_train_val', default=False, type=str2bool,
                        help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--out_num', default=10, type=int) 
parser.add_argument('--total_epoch', default=200, type=int, help='option to more than 200 epoch')
parser.add_argument('--resume', default=False, action='store_true', help='option to use cutout augmentation')

# augmentation type and parameters
parser.add_argument('--da', default=False, action='store_true', help='option to use augmentation')
parser.add_argument('--cutmix', default=False, action='store_true', help='option to use cutmix augmentation')
parser.add_argument('--mixup', default=False, action='store_true', help='option to use mixup augmentation')

# loss computation
parser.add_argument('--smooth_ratio', default=0.75, type=float,help='fix the mix ratio')
parser.add_argument('--self_mix', action='store_true', default=False, help="flag for semi-supervised learning")

parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

def main():
    # Load the parameters from json file
    args = parser.parse_args()

    print('args: {}'.format(args))

    json_path = os.path.join(args.model_dir, 'params.json')
    args.model_dir = args.model_dir + f'/split{args.split_idx}'
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils1.Params(json_path)

    # Set the random seed for reproducible experiments
    random.seed(230)
    torch.manual_seed(230)
    np.random.seed(230)
    torch.cuda.manual_seed(230)

    # Set the logger
    utils1.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    if args.dataset == 'imagenet':
        args.train_classes = range(1000)
        args.open_set_classes = range(1000)
    elif args.dataset == 'scars':
        args.train_classes = [1, 11, 25, 38, 46, 50, 53, 75, 84, 100, 105, 117, 123, 129, 133, 134, 135, 136, 137, 138, 140, 144, 145, 146, 147, 149, 150, 151, 153, 160, 161, 162, 163, 164, 167, 168, 169, 174, 175, 180, 185, 186, 187, 192, 193, 0, 81, 97, 104, 122, 139, 141, 142, 143, 148, 152, 154, 155, 156, 157, 158, 159, 165, 166, 170, 171, 172, 173, 176, 177, 181, 184, 188, 191, 194, 195, 2, 7, 9, 16, 20, 26, 28, 44, 54, 95, 98, 102, 127, 178, 182, 22, 41, 82, 93, 112, 125, 189]
        args.open_set_classes = [23, 42, 83, 94, 113, 126, 190, 3, 8, 10, 17, 21, 27, 29, 45, 55, 96, 99, 103, 128, 179, 183, 4, 5, 6, 12, 13, 14, 15, 18, 19, 24, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 43, 47, 48, 49, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 85, 86, 87, 88, 89, 90, 91, 92, 101, 106, 107, 108, 109, 110, 111, 114, 115, 116, 118, 119, 120, 121, 124, 130, 131, 132]
    else:
        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,
                                                                     cifar_plus_n=args.out_num)
    
    args.num_class = len(args.train_classes)
    print('train classes: {}'.format(args.train_classes))
    print('open set classses: {}'.format(args.open_set_classes))
    # ------------------------
    # DATASETS
    # ------------------------
    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                            open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                            split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                            args=args)


    # ------------------------
    # DATALOADER
    # ------------------------

    dataloaders = {}
    for k, v, in datasets.items():
        if 'train' in k:
            shuffle = True
        shuffle = True if k == 'train' else False
        dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                    shuffle=shuffle, sampler=None, num_workers=args.num_workers)
    train_dl = dataloaders['train']
    dev_dl = dataloaders['val']
    print(enumerate(train_dl))
    print(enumerate(dev_dl))

    """
    Load student and teacher model
    """
    if "distill" in params.model_version:

        # Specify the student models
        if params.model_version == "mobilenet_v2_distill":
            print("Student model: {}".format(params.model_version))
            model = mobilenet.mobilenetv2(class_num=args.num_class).cuda()

        elif params.model_version == 'resnet18_distill':
            print("Student model: {}".format(params.model_version))
            model = resnet.ResNet18(num_classes=args.num_class).cuda()
            # model = timm.create_model('resnet18', num_classes=20, pretrained=True)

        # optimizer
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate * (args.batch_size / 128), momentum=0.9,
                                weight_decay=5e-4)

        iter_per_epoch = len(train_dl)
        warmup_scheduler = utils1.WarmUpLR(optimizer,
                                          iter_per_epoch * args.warm)  # warmup the learning rate in the first epoch

        # specify loss function
        loss_fn_kd = loss_DIST

        """ 
            Specify the pre-trained teacher models for knowledge distillation
            Checkpoints can be obtained by regular training or downloading our pretrained models
            For model which is pretrained in multi-GPU, use "nn.DaraParallel" to correctly load the model weights.
        """
        if params.teacher == "resnet18":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = resnet.ResNet18(num_classes=args.num_class)
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet18/best.pth.tar'
            if args.pt_teacher:  # poorly-trained teacher for Defective KD experiments
                teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet18/0.pth.tar'
            teacher_model = teacher_model.cuda()
        
        elif params.teacher == "resnet50":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = resnet.ResNet50(num_classes=args.num_class).cuda()
            teacher_checkpoint = 'path/to/teacher_checkpoint'

        elif params.teacher == "resnet101":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = resnet.ResNet101(num_classes=args.num_class).cuda()
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet101/split{}/last.pth.tar'.format(args.split_idx)
            teacher_model = teacher_model.cuda()

        # utils1.load_checkpoint(teacher_checkpoint, teacher_model)

        # Train the model with KD
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        
        train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd,
                              warmup_scheduler, params, args, dataloaders, args.restore_file)
    # non-KD mode: regular training to obtain a baseline model
    else:
        print("Train base model")
        if params.model_version == "mobilenet_v2":
            print("model: {}".format(params.model_version))
            model = mobilenet.mobilenetv2(class_num=args.num_class).cuda()

        elif params.model_version == "resnet18":
            
            model = resnet.ResNet18(num_classes=args.num_class).cuda()
            if args.dataset == 'cub':
                model = models.resnet18(pretrained=False).cuda()

        elif params.model_version == "resnet50":
            model = resnet.ResNet50(num_classes=args.num_class).cuda()
            if args.dataset == 'cub':
                model = models.resnet50(pretrained=False).cuda()

        elif params.model_version == "resnet101":
            model = resnet.ResNet101(num_classes=args.num_class).cuda()

        loss_fn = nn.CrossEntropyLoss()
        
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate * (args.batch_size / 128), momentum=0.9,
                                  weight_decay=5e-4)

        iter_per_epoch = len(train_dl)
        warmup_scheduler = utils1.WarmUpLR(optimizer, iter_per_epoch * args.warm)

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, params,
                           args.model_dir, warmup_scheduler, args, dataloaders, args.restore_file)


if __name__ == '__main__':
    main()

