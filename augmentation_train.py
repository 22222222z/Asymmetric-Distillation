import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os 
from random import choice

device = None
if hasattr(os.environ, "LOCAL_RANK"):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda:{}".format(local_rank))

def rand_bbox(size, args):

    # sample lam from uniform distribution
    fix_ratio = args.fix_ratio
    lam = fix_ratio if fix_ratio != 0 else np.random.beta(1.0, 1.0)
    if args.ssl and lam < 0.5:
        lam = 1. - lam
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

    return lam, bbx1, bby1, bbx2, bby2

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

def mix_same_label(batch_size, labels, args):
    
    count = torch.tensor([0]*args.num_class).cuda()
    for i in range(batch_size):
        count[labels[i]] += 1
    # print(count)
    sorted_index = torch.zeros((20, batch_size))

    count_dict = {}
    for i in range(args.num_class):
        for j in range(batch_size):
            if labels[j] == i:
                idx = str(i)
                if idx in count_dict.keys():
                    count_dict[idx].extend([j])
                else:
                    count_dict.update({idx: [j]})

    rand_index = []
    labels = labels.cpu().detach().numpy().tolist()
    for i in range(batch_size):
        idx = str(labels[i])
        v = choice(count_dict[idx])
        rand_index.extend([v])
        count_dict[idx].remove(v)
    return rand_index

def mix_batch(train_batch, option, args, labels=None):

    fix_ratio = args.fix_ratio
    mix_batch = train_batch.clone()
    if args.self_mix and labels is not None:
        rand_index = mix_same_label(train_batch.size()[0], labels, args)
        rand_index = torch.tensor(rand_index).cuda()
    else:
        if not device:
            rand_index = torch.randperm(train_batch.size()[0]).cuda()
        else:
            rand_index = torch.randperm(train_batch.size()[0]).to(device)

    if option == 'cutmix':
        lam, bbx1, bby1, bbx2, bby2 = rand_bbox(train_batch.size(), args)
        mix_batch[:, :, bbx1:bbx2, bby1:bby2] = mix_batch[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (mix_batch.size()[-1] * mix_batch.size()[-2]))
    elif option == 'mixup':
        lam = fix_ratio if fix_ratio != 0 else np.random.beta(1.0, 1.0)
        if args.ssl and lam > 0.5:
            lam = 1. - lam
        mix_batch = mix_batch * lam + mix_batch[rand_index] * (1. - lam)

    return lam, rand_index, mix_batch

def mix_batch_u(base, patch, option, args, labels=None):
    fix_ratio = args.fix_ratio
    mix_batch = base.clone()
    rand_index = torch.randperm(mix_batch.size()[0]).cuda()

    if option == 'cutmix':
        lam, bbx1, bby1, bbx2, bby2 = rand_bbox(base.size(), args)
        mix_batch[:, :, bbx1:bbx2, bby1:bby2] = patch[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (mix_batch.size()[-1] * mix_batch.size()[-2]))
    elif option == 'mixup':
        lam = fix_ratio if fix_ratio != 0 else np.random.beta(1.0, 1.0)
        if args.ssl and lam > 0.5:
            lam = 1. - lam
        mix_batch = mix_batch * lam + mix_batch[rand_index] * (1. - lam)

    return lam, rand_index, mix_batch

def cutout_batch(train_batch, args):
    
    bbx1, bby1, bbx2, bby2 = rand_bbox_cutout(train_batch.size(), args.cutout_length)

    aug_batch = train_batch.clone()
    cutout_batch = torch.zeros(train_batch.size())
    aug_batch[:, :, bbx1:bbx2, bby1:bby2] = cutout_batch[:, :, bbx1:bbx2, bby1:bby2]

    return aug_batch

def verify_teacher(output_teacher_batch_cutmix, target_a, target_b, lam, args):

    prob = F.softmax(output_teacher_batch_cutmix, dim=-1)
    prob = prob.cpu().detach().numpy().tolist()

    num = 0
    learn_list = []
    uni_list = []
    p_label = []

    for jj in range(output_teacher_batch_cutmix.size()[0]):

        max_prob = max(prob[jj])
        min_prob = min(prob[jj])
        l_a = target_a[jj]
        l_b = target_b[jj]
        a_p = prob[jj][l_a]
        b_p = prob[jj][l_b]
        if a_p != max_prob and b_p != max_prob:
            uni_list.append(jj)
            ppp_label = []
            ppp_label.extend([1/args.num_class * args.smooth_ratio]*args.num_class)
            ppp_label[target_a[jj]] = ppp_label[target_a[jj]] + (1. - args.smooth_ratio) * lam
            ppp_label[target_b[jj]] = ppp_label[target_b[jj]] + (1. - args.smooth_ratio) * (1. - lam)

            ppp_sum = 0
            for i in ppp_label:
                ppp_sum = ppp_sum + i
            assert (ppp_sum < 1.1 and ppp_sum > 0.95), 'Relabel出错!'

            p_label.append(ppp_label)
            num = num + 1
        else:
            learn_list.append(jj)

    return learn_list, uni_list, p_label