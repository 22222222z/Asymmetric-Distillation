"""
Tensorboard logger code referenced from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/
Other helper functions:
https://github.com/cs230-stanford/cs230-stanford.github.io
"""

import json
import logging
import os
import shutil
import torch
from collections import OrderedDict
from torch.optim.lr_scheduler import _LRScheduler

from utils.utils import strip_state_dict
import torch.nn.functional as F
import math

#import tensorflow as tf
import numpy as np
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint, epoch_checkpoint = False):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    # if is_best:
    #     shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
    if epoch_checkpoint == True:
        epoch_file = str(state['epoch']-1) + '.pth.tar'
        shutil.copyfile(filepath, os.path.join(checkpoint, epoch_file))

def save_checkpoint_interval(epoch, state, is_best, checkpoint, epoch_checkpoint = False):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'epoch_{}.pth.tar'.format(epoch+1))
    print('saving epoch {} .pth.tar'.format(epoch + 1))
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
    if epoch_checkpoint == True:
        epoch_file = str(state['epoch']-1) + '.pth.tar'
        shutil.copyfile(filepath, os.path.join(checkpoint, epoch_file))





def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    print('checkpoitn:{}'.format(checkpoint))
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))

    # modified
    pthtar = False
    if '.pth.tar' in checkpoint:
        pthtar = True

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    # modified
    if pthtar:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        state_dict = strip_state_dict(checkpoint)
        model.load_state_dict(state_dict)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

'''
class Board_Logger(object):
    """Tensorboard log utility"""
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush
'''

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def rand_fixed_bbox(size, lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(cut_w // 2, W - cut_w // 2)
    cy = np.random.randint(cut_h // 2, H - cut_h // 2)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def find_second_max(list):
    idx = list.index(max(list))
    list[idx] = min(list)
    idx = list.index(max(list))
    return idx

def cutmix(train_batch, labels_batch, output_teacher_batch, args):
    # generate mixed sample
    lam = np.random.beta(args.beta, args.beta)
    
    # 随机混合
    rand_index = torch.randperm(train_batch.size()[0]).cuda()
    target_a = labels_batch
    target_b = labels_batch[rand_index]
    if args.print_num > 0:
        print('target_a: {}'.format(target_a))
        print('target_b: {}'.format(target_b))
        for ti in range(3):
            print(output_teacher_batch[ti])
    bbx1, bby1, bbx2, bby2 = rand_bbox(train_batch.size(), lam)
    train_batch_cutmix = train_batch.clone()
    train_batch_cutmix[:, :, bbx1:bbx2, bby1:bby2] = train_batch_cutmix[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (train_batch_cutmix.size()[-1] * train_batch_cutmix.size()[-2]))
    if args.print_num > 0:
        print('lamda: {}'.format(lam))

    return lam, train_batch_cutmix, target_a, target_b, rand_index

def rand_bbox_cutout(size, length):
    W = size[2]
    H = size[3]

    cut_w = np.int_(length)
    cut_h = np.int_(length)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def rand_bbox_mosaic(size, lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    # cx = np.random.randint(cut_w // 2, W - cut_w // 2)
    # cy = np.random.randint(cut_h // 2, H - cut_h // 2)

    # bbx1 = np.clip(cx - cut_w // 2, 0, W)
    # bby1 = np.clip(cy - cut_h // 2, 0, H)
    # bbx2 = np.clip(cx + cut_w // 2, 0, W)
    # bby2 = np.clip(cy + cut_h // 2, 0, H)

    return cut_w, cut_h

def entropy(Plist):
    if len(Plist):
        result=0
        for x in Plist:
            result+=(-x)*math.log(x,2)
    else:
        print('Error loading data')
        result=-1
    return result


def compute_top(p):
    p = F.softmax(p, dim=-1)
    p = p.cpu().detach().numpy().tolist()
    bs = len(p)
    idx = []
    h = []
    for i in range(bs):
        h.append(entropy(p[i]))

    K = bs // 2
    for _ in range(K):
        max_val = max(h)
        max_idx = h.index(max_val)

        # Nmax_val_lis.append(max_val)
        idx.append(max_idx)

        h[max_idx] = float('-inf')
    
    return idx

def threshold_cutmix(output, thre, target):
    prob = F.softmax(output, dim=-1)
    idx = []
    for i in range(output.size(0)):
        iidx = torch.argmax(output[i])
        if prob[i][iidx] > thre and iidx==target[i]:
            idx.append(i)

    return idx

def threshold_cutmix_unlabeled(output, thre):
    prob = F.softmax(output, dim=-1)
    idx = []
    for i in range(output.size(0)):
        iidx = torch.argmax(output[i])
        if prob[i][iidx] > thre:
            idx.append(i)

    return idx

def threshold_cutmix_lower(output, thre, args):
    # target.extend(target)
    prob = F.softmax(output, dim=-1)
    idx = []
    for i in range(output.size(0)):
        iidx = torch.argmax(output[i])
        if prob[i][iidx] < thre:
            idx.append(i)
    # uni_label = []
    # uuni_label = [1/args.num_class * 0.75]*args.num_class
    # uuni_label = torch.ones((1, args.num_class)) / args.num_class
    uuni_label = torch.ones((1, args.num_class)) / args.num_class
    uni_label = torch.ones((1, args.num_class)) / args.num_class
    # uni_label.cuda()
    # uuni_label.cuda()
    if len(idx) == 1 or len(idx) == 0:
        uni_label.unsqueeze(0)
    else:
        for i in range(1, len(idx)):
            # uni_label.append(uuni_label)
            uni_label = torch.cat((uni_label, uuni_label), dim=0)

    # uni_label = []
    # uuni_label = [1/args.num_class * 0.75]*args.num_class
    # for i in range(len(idx)):
        # num += 1
        # uni_label.append(uuni_label)
    # if num == 1:
    #     uni_label = [uni_label]

    # return idx, uni_label

    return idx, uni_label

def threshold_single(output_teacher, target_a, target_b):

    thre = 0.8
    prob = F.softmax(output_teacher, dim=-1)
    idx_a = []
    idx_b = []
    for i in range(output_teacher.size(0)):
        iidx = torch.argmax(output_teacher[i])
        if prob[i][iidx] > thre and iidx == target_a[i]:
            idx_a.append(i)
        elif prob[i][iidx] > thre and iidx == target_b[i]:
            idx_b.append(i)
    
    return idx_a, idx_b

