import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)

def inter_class_relation(y_s, y_t, lamada=None):
    if lamada == None:
        # print('22222222222222222222222222222222')
        return 1 - pearson_correlation(y_s, y_t).mean()
    else:
        # print('11111111111111111111111111111111')
        return 1 - (pearson_correlation(y_s, y_t) * lamada).mean()


def intra_class_relation(y_s, y_t, lamada=None):
    if lamada == None:
        return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))
    else:
        return inter_class_relation((y_s * lamada.unsqueeze(-1)).transpose(0, 1), (y_t * lamada.unsqueeze(-1)).transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t, lamada=None, ts=False):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        if ts:
            y_t = z_t   
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t, lamada)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss

        return kd_loss

# --DIST
def loss_DIST(outputs, labels, teacher_outputs, params):
    """
    loss function for Knowledge Distillation (KD) and CrossEntropy
    params.alpha: weight for cls loss
    """
    dist = DIST(beta=params.beta, gamma=params.gamma, tau=1)
    loss_dist = dist(outputs, teacher_outputs)

    loss_CE = F.cross_entropy(outputs, labels)
    # alpha = params.alpha
    # if alpha < 1.0:
    #     return alpha * loss_CE + (1. - alpha) * loss_dist
    # else:
    #     return alpha * loss_CE + loss_dist
    
    return params.alpha * loss_CE + loss_dist

# --DA
def loss_DA(outputs, teacher_outputs, params, lamada=None, stu_feat=None, tea_feat=None):
    """
    loss function for Knowledge Distillation (KD) w/o hard label
    """
    dist = DIST(beta=params.beta, gamma=params.gamma, tau=1)
    loss_dist = dist(outputs, teacher_outputs, lamada)

    return loss_dist