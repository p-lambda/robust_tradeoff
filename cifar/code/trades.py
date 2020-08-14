import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

import numpy as np

from spatial_attack_cifar10 import get_batch_spatial_adv_example


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean',
                  smooth_eps=None, smooth_dist=None, from_logits=True):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1. - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, weight=None, ignore_index=-1, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True):
        super(CrossEntropyLoss, self).__init__(weight=weight,
                                               ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction, smooth_eps=self.smooth_eps,
                             smooth_dist=smooth_dist, from_logits=self.from_logits)


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def trades_loss(model,
                epoch,
                x_natural,
                y,
                unsup,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf',
                batch_mode='joint',
                freeze_natural=False,
                extra_class_weight=None,
                entropy_weight=0,
                unlabeled_natural_weight=0.0,
                unlabeled_robust_weight=0.0,
                init='inside'):
    if beta == 0:
        logits = model(x_natural)
        loss = F.cross_entropy(logits, y)
        inf = torch.Tensor([np.inf])
        zero = torch.Tensor([0.])
        return loss, loss, inf, inf, zero

    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.  # the + 0. is for copying the tensor
    if distance == 'l_inf':
        if init == 'inside':
            x_adv += 0.001 * torch.randn(x_natural.shape).cuda().detach()
        elif init == 'boundary':
            x_adv += epsilon * torch.randn(x_natural.shape).sign().cuda().detach()
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise ValueError('Unknown attack initialization %s' % init)

        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon),
                              x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    # zero gradient
    optimizer.zero_grad()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    if batch_mode == 'clean':
        logits = model(x_natural)
        with _disable_tracking_bn_stats(model):
            logits_adv = F.log_softmax(model(x_adv), dim=1)

    elif batch_mode == 'adv':
        # NOTE: the fact that we're computing a forward pass with x_adv here
        # means that the batch normalization statistics update with the perturbed
        # input. For l_inf this is (probably) not very meaningful, as the
        # perturbation is small. However, for l_2_rand this is meaningful, as the
        # perturbation is typically large. Empirically, the l_2_rand training
        # breaks down completely if we do not let batch norm update the statistics
        # with the perturbation. However, this means that to recover clean training
        # with l_2_rand, it is not sufficient to set beta=0; one must also set
        # epsilon to be small.
        logits_adv = F.log_softmax(model(x_adv), dim=1)
        with _disable_tracking_bn_stats(model):
            logits = model(x_natural)

    elif batch_mode == 'joint':
        logits_joint = F.log_softmax(model(torch.cat([x_natural, x_adv])),
                                     dim=-1)
        logits, logits_adv = torch.split(logits_joint, len(x_natural))

    elif batch_mode == 'both':
        logits_adv = F.log_softmax(model(x_adv), dim=1)
        logits = model(x_natural)
    else:
        raise ValueError('Unknown batch mode %s' % batch_mode)

    if freeze_natural:
        with torch.no_grad():
            p_natural = F.softmax(logits, dim=1)
            assert (
                p_natural.requires_grad == False), "Natural logits not frozen"
    else:
        p_natural = F.softmax(logits, dim=1)

    # Splitting robust loss into labeled and unlabeled
    is_labeled = unsup != 1
    is_unlabeled = unsup == 1
    if is_unlabeled.sum().float() > 0:

        num_labeled = is_labeled.sum().float()
        num_unlabeled = is_unlabeled.sum().float()

        loss_robust_labeled = (1.0 / is_labeled.sum().float()) * criterion_kl(
            logits_adv[is_labeled], p_natural[is_labeled])
        loss_robust_unlabeled = (1.0 / is_unlabeled.sum().float()) * criterion_kl(
            logits_adv[is_unlabeled], p_natural[is_unlabeled])

        loss_natural_labeled = F.cross_entropy(logits[is_labeled], y[is_labeled], ignore_index=-1, reduction='sum')
        loss_natural_unlabeled = F.cross_entropy(logits[is_unlabeled], y[is_unlabeled], ignore_index=-1, reduction='sum')

        loss_robust = ((1 - unlabeled_robust_weight)*loss_robust_labeled +
                       unlabeled_robust_weight*loss_robust_unlabeled)/((1 - unlabeled_robust_weight)*num_labeled + unlabeled_robust_weight*num_unlabeled)
        loss_natural = ((1 - unlabeled_natural_weight)*loss_natural_labeled +
                       unlabeled_natural_weight*loss_natural_unlabeled)/((1 - unlabeled_natural_weight)*num_labeled + unlabeled_natural_weight*num_unlabeled)

    else:
        loss_natural = F.cross_entropy(logits, y, ignore_index=-1)
        loss_robust = (1.0 / batch_size) * criterion_kl(logits_adv, p_natural)
    loss = (1-beta) *loss_natural + beta * loss_robust

    # dummy
    loss_entropy_unlabeled = torch.tensor(0)

    return loss, loss_natural, loss_robust, loss_robust_labeled, loss_entropy_unlabeled


# Changed the step_size to what the Madry paper has
def madry_loss(model,
               epoch,
               x_natural,
               y,
               unsup,
               optimizer,
               step_size=0.007,
               epsilon=0.031,
               perturb_steps=10,
               beta=1.0,
               unlabeled_natural_weight=1.0,
               unlabeled_robust_weight=0.0,
               distance='l_inf',
               batch_mode='joint',
               freeze_natural=False,
               rand_init=True,
               extra_class_weight=None,
               max_rot=30,
               max_trans=0.1071,
               soft_labels=False):
    # normalize the y
    if soft_labels:
        y = F.softmax(y, dim=1)

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_detached = x_natural.detach()
    x_adv = x_detached + 0.  # the + 0. is for copying the tensor
    if distance != 'l_inf' and soft_labels:
        raise ValueError("Soft labels only supported for l_inf")

    if distance == 'l_inf':
        if rand_init:
            x_adv += 0.001 * torch.randn(x_natural.shape).cuda().detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if soft_labels:
                    loss_adv = CrossEntropyLoss()(model(x_adv), y)
                else:
                    loss_adv = nn.CrossEntropyLoss()(model(x_adv), y)

            grad = torch.autograd.grad(loss_adv, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon),
                              x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'spatial':
        x_adv = get_batch_spatial_adv_example(model, x_natural, y, max_rot, max_trans, random=False, wo10=True)
    elif distance == 'spatial_random':
        x_adv = get_batch_spatial_adv_example(model, x_natural, y, max_rot, max_trans, random=True)
    else:
        raise ValueError("distance not supported")

    model.train()
    # zero gradient
    optimizer.zero_grad()
    # Freezing x_adv
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    if batch_mode == 'clean':
        logits = model(x_natural)
        with _disable_tracking_bn_stats(model):
            logits_adv = model(x_adv)

    elif batch_mode == 'adv':
        logits_adv = model(x_adv)
        with _disable_tracking_bn_stats(model):
            logits = model(x_natural)

    elif batch_mode == 'joint':
        logits_joint = model(torch.cat([x_natural, x_adv]))
        logits, logits_adv = torch.split(logits_joint, len(x_natural))

    else:
        logits_adv = model(x_adv)
        logits = model(x_natural)

    # Splitting robust loss into labeled and unlabeled
    is_labeled = unsup != 1
    is_unlabeled = unsup == 1

    if is_unlabeled.sum().float() > 0:
        num_labeled = is_labeled.sum().float()
        num_unlabeled = is_unlabeled.sum().float()
        if soft_labels:
            # technically this also can use regular targets, but we don't change the original code
            loss_robust_labeled = CrossEntropyLoss(ignore_index=-1, reduction='sum')(logits_adv[is_labeled], y[is_labeled])
            loss_robust_unlabeled = CrossEntropyLoss(ignore_index=-1, reduction='sum')(logits_adv[is_unlabeled], y[is_unlabeled])
            loss_natural_labeled = CrossEntropyLoss(ignore_index=-1, reduction='sum')(logits[is_labeled], y[is_labeled])
            loss_natural_unlabeled = CrossEntropyLoss(ignore_index=-1, reduction='sum')(logits[is_unlabeled], y[is_unlabeled])

        else:
            loss_robust_labeled = F.cross_entropy(
                logits_adv[is_labeled], y[is_labeled], ignore_index=-1, reduction='sum')
            loss_robust_unlabeled = F.cross_entropy(
                logits_adv[is_unlabeled], y[is_unlabeled], ignore_index=-1, reduction='sum')
            loss_natural_labeled = F.cross_entropy(
                logits[is_labeled], y[is_labeled], ignore_index=-1, reduction='sum')
            loss_natural_unlabeled = F.cross_entropy(
                logits[is_unlabeled], y[is_unlabeled], ignore_index=-1, reduction='sum')

        loss_robust = ((1 - unlabeled_robust_weight)*loss_robust_labeled +
                       unlabeled_robust_weight*loss_robust_unlabeled)/((1 - unlabeled_robust_weight)*num_labeled + unlabeled_robust_weight*num_unlabeled)
        loss_natural = ((1 - unlabeled_natural_weight)*loss_natural_labeled +
                       unlabeled_natural_weight*loss_natural_unlabeled)/((1 - unlabeled_natural_weight)*num_labeled + unlabeled_natural_weight*num_unlabeled)

    else:
        if soft_labels:
            loss_natural = CrossEntropyLoss(ignore_index=-1)(logits, y)
            loss_robust = CrossEntropyLoss(ignore_index=-1)(logits_adv, y)
        else:
            loss_natural = F.cross_entropy(logits, y, ignore_index=-1)
            loss_robust = F.cross_entropy(logits_adv, y, ignore_index=-1)

    loss = (1 - beta)*loss_natural + beta * loss_robust

    return loss, loss_natural, loss_robust, loss_natural_labeled, loss_natural_unlabeled
