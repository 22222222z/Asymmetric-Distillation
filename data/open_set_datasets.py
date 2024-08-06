from data.cifar import get_cifar_10_10_datasets, get_cifar_10_100_datasets
from data.tinyimagenet import get_tiny_image_net_datasets
from data.svhn import get_svhn_datasets
from data.mnist import get_mnist_datasets
from data.cub import get_cub_datasets
from data.stanford_cars import get_scars_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.pku_aircraft import get_pku_aircraft_datasets

from data.open_set_splits.osr_splits import osr_splits
from data.augmentations import get_transform
from config import osr_split_dir

import os
import sys
import pickle
import torch

"""
For each dataset, define function which returns:
    training set
    validation set
    open_set_known_images
    open_set_unknown_images
"""

get_dataset_funcs = {
    'cifar-10-100': get_cifar_10_100_datasets,
    'cifar-10-10': get_cifar_10_10_datasets,
    'mnist': get_mnist_datasets,
    'svhn': get_svhn_datasets,
    'tinyimagenet': get_tiny_image_net_datasets,
    'tinyimagenet_full': get_tiny_image_net_datasets,
    'cub': get_cub_datasets,
    'scars': get_scars_datasets,
    'aircraft': get_aircraft_datasets,
    'pku-aircraft': get_pku_aircraft_datasets
}

def get_datasets(name, transform='default', image_size=224, train_classes=(0, 1, 8, 9),
                 open_set_classes=range(10), balance_open_set_eval=False, split_train_val=True, seed=0, args=None):

    """
    :param name: Dataset name
    :param transform: Either tuple of train/test transforms or string of transform type
    :return:
    """

    print('Loading datasets...')

    if isinstance(transform, tuple):
        train_transform, test_transform = transform
    else:
        train_transform, test_transform = get_transform(transform_type=transform, image_size=image_size, args=args)

    if name in get_dataset_funcs.keys():
        datasets = get_dataset_funcs[name](train_transform, test_transform,
                                train_classes=train_classes,
                                open_set_classes=open_set_classes,
                                balance_open_set_eval=balance_open_set_eval,
                                split_train_val=split_train_val,
                                seed=seed)
    else:
        raise NotImplementedError

    return datasets

def get_class_splits(dataset, split_idx=0, cifar_plus_n=10):

    if dataset in ('cifar-10-10', 'mnist', 'svhn'):
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(10) if x not in train_classes]

    elif dataset == 'cifar-10-100':
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = osr_splits['cifar-10-100-{}'.format(cifar_plus_n)][split_idx]

    elif dataset == 'tinyimagenet':
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(200) if x not in train_classes]
    
    elif dataset == 'tinyimagenet_full':
        train_classes = [x for x in range(200)]
        open_set_classes = []

    elif dataset == 'cub':

        osr_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
        with open(osr_path, 'rb') as f:
            class_info = pickle.load(f)

        train_classes = class_info['known_classes']

        open_set_classes = class_info['unknown_classes']
        open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

    elif dataset == 'aircraft':

        osr_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
        with open(osr_path, 'rb') as f:
            class_info = pickle.load(f)

        train_classes = class_info['known_classes']

        open_set_classes = class_info['unknown_classes']
        open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

    elif dataset == 'pku-aircraft':
        print('Warning: PKU-Aircraft dataset has only one open-set split')
        train_classes = list(range(180))
        open_set_classes = list(range(120))

    else:

        raise NotImplementedError
    
    # print('train_classes:',train_classes)
    # if dataset == 'cub':
    #     cub_classes = [150, 70, 34, 178, 199, 131, 129, 147, 134, 11, 26, 93, 95, 121, 123, 99, 149, 167, 18, 31, 69, 198, 116, 158, 126, 17, 5, 179, 111, 163, 184, 81, 174, 42, 53, 89, 77, 55, 23, 48, 43, 44, 56, 28, 193, 143, 0, 176, 84, 15, 38, 154, 141, 190, 172, 124, 189, 19, 80, 157, 12, 9, 79, 30, 94, 67, 197, 97, 168, 137, 119, 76, 98, 88, 40, 106, 171, 87, 166, 186, 27, 51, 144, 135, 161, 64, 177, 7, 146, 61, 50, 162, 133, 82, 39, 74, 72, 91, 196, 136]
    #     if len(cub_classes) == len(train_classes):
    #         flag = True
    #         for i in range(len(cub_classes)):
    #             if cub_classes[i] != train_classes[i]:
    #                 flag = False
    #         if flag:
    #             print('{}: Same split with split0'.format(split_idx))
    #             exit()

    return train_classes, open_set_classes

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__