'''
Adapted from TRADES code: https://github.com/yaodongyu/TRADES
'''
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from utils import get_model
from trades import trades_loss, madry_loss
from datasets import SemiSupervisedDataset, SemiSupervisedSampler, DATASETS
from pgd_attack_cifar10 import pgd_whitebox
from spatial_attack_cifar10 import pgd_whitebox_spatial
from augmentation import LinfAug

import logging


parser = argparse.ArgumentParser(
    description='PyTorch TRADES Adversarial Training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=DATASETS,
                    help='The dataset to use for training)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='cancels the run if an appropriate checkpoint is found')
parser.add_argument('--weight_decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr_schedule', type=str, default='original',
                    choices=('v0', 'v0f', 'v1', 'cosine', 'wrn'),
                    help='learning rate schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--max_rot', default=30, type=int,
                    help='max rotation angle degrees')
parser.add_argument('--max_trans', default=0.1071, type=float,
                    help='max trans pixels (fraction of img dim)')
parser.add_argument('--epsilon_schedule', default=None, type=str,
                    help='schedule for increasing epsilon parameter. '
                         'rampXX means linearly increasing epsilon for XX% of'
                         ' the training, and then keeping it constant as '
                         'args.epsilon')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size', type=float)
parser.add_argument('--attack_init', default='inside', choices=('inside', 'boundary'),
                    help='initialization for adversarial perturbation during training', type=str)
parser.add_argument('--beta', default=1.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--take_amount_seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--data_dir', default='data', type=str,
                    help='directory where datasets are located')
parser.add_argument('--save_freq', '-s', default=50, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--eval_freq', '-evf', default=1, type=int,
                    help='eval frequency (in epochs)')
parser.add_argument('--train_eval_batches', default=None, type=int,
                    help='maximum number for batches in training set eval')
parser.add_argument('--train_downsample', '-td', default=1, type=int,
                    metavar='N',
                    help='how much to downsample training data')
parser.add_argument('--train_take_fraction', default=None, type=float,
                    metavar='N',
                    help='what fraction of labeled training samples to use')
parser.add_argument('--aux_take_amount', default=None, type=int,
                    metavar='N',
                    help='how much random aux examples to retain')
parser.add_argument('--semisup', action='store_true', default=False,
                    help='treats training data removed in downsampling as '
                         'unlabeled data (with label -1)')
parser.add_argument('--freeze_natural', action='store_true', default=False,
                    help='Whether to freeze natural logits for robust loss')
parser.add_argument('--unsup_fraction', '-uf', default=0.5, type=float,
                    metavar='f',
                    help='fraction of unlabeled examples in each batch (only '
                         'relevant with the --semisup flag)')
parser.add_argument('--model', '-m', default='wrn-40-2', type=str,
                    help='name of the model')
parser.add_argument('--eval_attack_batches', '-eab', default=5, type=int,
                    help='number of eval batches to attack with PGD')
parser.add_argument('--sup_labels', '-sl', default=None, type=int, nargs='+',
                    help='Labels to be used in supervised data')
parser.add_argument('--unsup_labels', '-ul', default=None, type=int, nargs='+',
                    help='Labels to be used in unsupervised data')
parser.add_argument('--test_labels', '-tl', default=None, type=int, nargs='+',
                    help='Labels to be used for clean accuracy evaluation')
parser.add_argument('--distance', '-d', default='l_inf', type=str,
                    help='metric for attack model.',
                    choices=['l_inf', 'spatial', 'spatial_random'])
parser.add_argument('--batch_mode', '-bm', default='both',
                    help='Which part of batches to update the statistics',
                    type=str, choices=('adv', 'clean', 'joint', 'both'))
parser.add_argument('--add_cifar100', action='store_true', default=False,
                    help='Adds all of CIFAR-100 as unlabeled data')
parser.add_argument('--no_aug', action='store_true', default=False,
                    help='Dont do any augmentation')
parser.add_argument('--unlabeled_class_weight', '-ucw', default=None,
                    type=float,
                    metavar='f',
                    help='weight of loss for predicting that an example was'
                         ' unlabeled (value of 0 will mean adding a dummy class')
parser.add_argument('--loss', default='trades', type=str,
                    choices=('trades', 'madry', 'noise', 'vat'),
                    help='which loss to use: trades (default) or madry '
                         '(adversarial training)'
                         'or noise (augmentation with noise)')
parser.add_argument('--aux_data_filename', default=None, type=str,
                    help='path to pickle file containing unlabeled data')
parser.add_argument('--aux_targets_filename', default=None, type=str,
                    help='path to pickle file containing labels of unlabeled data')
parser.add_argument('--add_aux_labels', action='store_true', default=False,
                    help='The infamous "approach 0"')
parser.add_argument('--aux_label_noise', default=None, type=float,
                    metavar='f',
                    help='fraction of aux labels to randomly permute')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='Use extragrdient steps')
parser.add_argument('--entropy_weight', type=float,
                    default=0.0, help='Weight on entropy loss')
parser.add_argument('--normalize_input', action='store_true', default=False,
                    help='Apply standard CIFAR normalization first thing '
                         'in the network (as part of the model, not in the data'
                         ' fetching pipline)')
parser.add_argument('--unlabeled_weight_schedule', default='step', type=str,
                    help='Schedule to use on the unlabeled examples (step or linear)')
parser.add_argument('--epoch_max_unlabeled_weight', default=0, type=int,
                    help='Epoch at which the unlabeled weight is at its maximum value of 1')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                    help='regularization coefficient (default: 0.01)')
parser.add_argument('--unlabeled_robust_weight', type=float, default=0.0, metavar='URW',
                    help='Weight on robust loss of unlabeled data')
parser.add_argument('--unlabeled_natural_weight', type=float, default=0.5, metavar='UNW',
                    help='Weight on natural loss of unlabeled data')

# What we had earlier as default is still maintained
args = parser.parse_args()

# should provide some improved performance
cudnn.benchmark = True

model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.model_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('TRADES semi-supervised training')
logging.info('Args: %s', args)

if not args.overwrite:
    final_checkpoint_path = os.path.join(
        model_dir, 'checkpoint-epoch{}.pt'.format(args.epochs))
    if os.path.exists(final_checkpoint_path):
        logging.info('Appropriate checkpoint found - quitting!')
        sys.exit(0)

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

if args.no_aug:
    # Override
    transform_train = transforms.ToTensor()

transform_test = transforms.Compose([
    transforms.ToTensor()])

trainset = SemiSupervisedDataset(base_dataset=args.dataset,
                                 downsample=args.train_downsample,
                                 take_fraction=args.train_take_fraction,
                                 semisupervised=args.semisup,
                                 add_cifar100=args.add_cifar100,
                                 add_svhn_extra=False,
                                 sup_labels=args.sup_labels,
                                 unsup_labels=args.unsup_labels,
                                 root=args.data_dir, train=True,
                                 download=True, transform=transform_train,
                                 aux_data_filename=args.aux_data_filename,
                                 aux_targets_filename=args.aux_targets_filename,
                                 add_aux_labels=args.add_aux_labels,
                                 aux_label_noise=args.aux_label_noise,
                                 aux_take_amount=args.aux_take_amount,
                                 take_amount_seed=args.take_amount_seed)

if args.semisup or args.add_cifar100 or (args.aux_data_filename is not None):
    # the repeat option makes sure that the number of gradient steps per 'epoch'
    # is roughly the same as the number of gradient steps in an epoch over full
    # CIFAR-10
    train_batch_sampler = SemiSupervisedSampler(
        trainset.sup_indices, trainset.unsup_indices,
        args.batch_size, args.unsup_fraction,
        num_batches=int(np.ceil(50000 / args.batch_size)))
else:
    assert (-1 not in trainset.targets)
    train_batch_sampler = SemiSupervisedSampler(trainset.sup_indices, [],
                                                batch_size=args.batch_size,
                                                unsup_fraction=0.0,
                                                num_batches=int(np.ceil(
                                                    50000 / args.batch_size)))
epoch_size = len(train_batch_sampler) * args.batch_size

train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler, **kwargs)

if args.test_labels is None:
    args.test_labels = args.sup_labels

testset = SemiSupervisedDataset(base_dataset=args.dataset,
                                root=args.data_dir, train=False,
                                download=True, test_labels=args.test_labels,
                                transform=transform_test)

test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                         shuffle=False, **kwargs)

trainset_eval = SemiSupervisedDataset(
    base_dataset=args.dataset,
    add_svhn_extra=False,
    take_fraction=args.train_take_fraction,
    take_amount_seed=args.take_amount_seed,
    downsample=args.train_downsample, semisupervised=False,
    root=args.data_dir, train=True,
    sup_labels=args.sup_labels,
    unsup_labels=None,
    download=True, transform=transform_train)

eval_train_loader = DataLoader(trainset_eval, batch_size=args.test_batch_size,
                               shuffle=True, **kwargs)

eval_test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                              shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_metrics = []
    unlabeled_robust_weight = adjust_unlabeled_robust_weight(epoch)
    epsilon = adjust_epsilon(epoch)
    #logger.info('Using unlabeled robust weight of %g' % unlabeled_robust_weight)
    for batch_idx, (data, target, unsup) in enumerate(train_loader):
        data, target, unsup = data.to(device), target.to(device), unsup.to(device)
        optimizer.zero_grad()

        # calculate robust loss
        if args.loss == 'trades':
            (loss, natural_loss, robust_loss,
             robust_loss_labeled,
             entropy_loss_unlabeled) = trades_loss(
                 model=model,
                 epoch=epoch,
                 unsup=unsup,
                 x_natural=data,
                 y=target,
                 optimizer=optimizer,
                 step_size=args.step_size,
                 epsilon=epsilon,
                 perturb_steps=args.num_steps,
                 beta=args.beta,
                 distance=args.distance,
                 batch_mode=args.batch_mode,
                 extra_class_weight=args.unlabeled_class_weight,
                 entropy_weight=args.entropy_weight,
                 unlabeled_natural_weight=args.unlabeled_natural_weight,
                 unlabeled_robust_weight=unlabeled_robust_weight,
                 init=args.attack_init)
        elif args.loss == 'madry':
            assert (not args.semisup), 'Cannot use unsupervised data with Madry loss'
            assert (args.beta <= 1), 'Madry loss requires beta < 1'
            (loss, natural_loss, robust_loss,
             loss_natural_labeled,
             loss_natural_unlabeled) = madry_loss(
                model=model,
                epoch=epoch,
                x_natural=data,
                y=target,
                unsup=unsup,
                optimizer=optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta,
                unlabeled_natural_weight=args.unlabeled_natural_weight,
                unlabeled_robust_weight=args.unlabeled_robust_weight,
                distance=args.distance,
                batch_mode=args.batch_mode,
                extra_class_weight=args.unlabeled_class_weight,
                max_rot=args.max_rot,
                max_trans=args.max_trans)
            robust_loss_labeled = loss_natural_labeled
            entropy_loss_unlabeled = loss_natural_unlabeled

        loss.backward()
        optimizer.step()

        curr_train_metrics = dict(
            epoch=epoch,
            loss=loss.item(),
            natural_loss=natural_loss.item(),
            robust_loss=robust_loss.item(),
            robust_loss_labeled=robust_loss_labeled.item(),
            entropy_loss_unlabeled=entropy_loss_unlabeled.item())
        train_metrics.append(curr_train_metrics)

        # print progress
        if batch_idx % args.log_interval == 0:
            logging.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), epoch_size,
                           100. * batch_idx / len(train_loader), loss.item()))
    return train_metrics


def eval(args, model, device, eval_set, loader, epoch=None):
    loss = 0
    total = 0
    correct = 0
    adv_correct = 0
    adv_correct_clean = 0
    adv_total = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, unsup) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data, target = data[target != -1], target[target != -1]
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx < args.eval_attack_batches:
                if args.distance == 'l_inf':
                    incorrect_clean, incorrect_rob = (
                        o.item() for o in pgd_whitebox(
                        model, data, target,
                        epsilon=args.epsilon,
                        num_steps=2 * args.num_steps,
                        step_size=args.step_size))
                elif args.distance in {'spatial', 'spatial_random'}:
                    use_random = (args.distance == 'spatial_random')
                    incorrect_clean, incorrect_rob = (
                        o.item() for o in pgd_whitebox_spatial(
                            model, data, target,
                            max_rot=args.max_rot,
                            max_trans=args.max_trans,
                            random=use_random))
                else:
                    raise ValueError('No support for distance %s',
                                     args.distance)
                adv_correct_clean += (len(data) - int(incorrect_clean))
                adv_correct += (len(data) - int(incorrect_rob))
                adv_total += len(data)
            total += len(data)
            if ((eval_set == 'train') and
                    (batch_idx + 1 == args.train_eval_batches)):
                break
    loss /= total
    accuracy = correct / total
    if adv_total > 0:
        # measuring robust clean accuracy is useful for randomized smoothing
        # because it tells us how much damage the randomized smoothing does
        # for PGD, this should be the same as accuracy but will not be identical
        # since we only compute it on a smaller subsample
        robust_clean_accuracy = adv_correct_clean / adv_total
        robust_accuracy = adv_correct / adv_total
    else:
        robust_accuracy = robust_clean_accuracy = 0.

    eval_data = dict(loss=loss, accuracy=accuracy,
                     robust_accuracy=robust_accuracy,
                     robust_clean_accuracy=robust_clean_accuracy)
    eval_data = {eval_set + '_' + k: v for k, v in eval_data.items()}
    eval_data.update({'epoch': epoch})
    # wandb.log(eval_data)
    logging.info(
        '{}: Clean loss: {:.4f}, '
        'Clean accuracy: {}/{} ({:.2f}%), '
        '{} clean accuracy: {}/{} ({:.2f}%), '
        'Robust accuracy {}/{} ({:.2f}%)'.format(
            eval_set.upper(), loss,
            correct, total, 100.0 * accuracy,
            'PGD',
            adv_correct_clean, adv_total, 100.0 * robust_clean_accuracy,
            adv_correct, adv_total, 100.0 * robust_accuracy))

    return eval_data


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    schedule = args.lr_schedule
    if schedule == 'v0':  # buggy schedule from TRADES repo
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
    elif schedule == 'v0f':
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
        if epoch >= 0.9 * args.epochs:
            lr = args.lr * 0.01
        if epoch >= args.epochs:
            lr = args.lr * 0.001
    elif schedule == 'v1':
        if epoch >= 0.5 * args.epochs:
            lr = args.lr * 0.2
    elif schedule == 'cosine':
        lr = args.lr * 0.5 * (1 + np.cos((epoch - 1) / args.epochs * np.pi))
    elif schedule == 'wrn':
        if epoch >= 0.3 * args.epochs:
            lr = args.lr * 0.2
    else:
        raise ValueError('Unkown LR schedule %s' % schedule)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_unlabeled_robust_weight(epoch):
    if args.unlabeled_weight_schedule == 'step':
        if epoch < args.epoch_max_unlabeled_weight:
            unlabeled_robust_weight = 0
        else:
            unlabeled_robust_weight = 1
    elif args.unlabeled_weight_schedule == 'linear':
        weights = np.linspace(0, 1,
                              int(args.epoch_max_unlabeled_weight))
        if epoch < int(args.epoch_max_unlabeled_weight):
            unlabeled_robust_weight = weights[epoch]
        else:
            unlabeled_robust_weight = 1
    else:
        unlabeled_robust_weight = 1
    return unlabeled_robust_weight


def adjust_epsilon(epoch):
    base_eps = args.epsilon
    if args.epsilon_schedule is None:
        return base_eps
    if args.epsilon_schedule.startswith('ramp'):
        cutoff_frac = float(args.epsilon_schedule[len('ramp'):]) / 100
        cutoff_epoch = args.epochs * cutoff_frac
        eps = min(base_eps, base_eps * (epoch-1) / cutoff_epoch)
        logger.info('Setting epsilon to %g' % eps)
        return eps
    raise ValueError('Unknown epsilon schedule %s' % args.epsilon_schedule)


def main():
    # init model, ResNet18() can be also used here for training
    train_df = pd.DataFrame()
    eval_df = pd.DataFrame()

    if args.sup_labels is None:
        num_classes = 10
    else:
        num_classes = len(args.sup_labels)

    if args.unlabeled_class_weight is not None:
        num_classes += 1

    model = get_model(args.model, num_classes=num_classes,
                      normalize_input=args.normalize_input)
    if use_cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)

    # save initial checkpoint
    torch.save(model.state_dict(),
               os.path.join(model_dir,
                            'initial.pt'))

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        lr = adjust_learning_rate(optimizer, epoch)
        logger.info('Setting learning rate to %g' % lr)
        # adversarial training
        train_data = train(args, model, device, train_loader, optimizer, epoch)
        train_df = train_df.append(pd.DataFrame(train_data), ignore_index=True)

        # evaluation on natural examples
        logging.info(120 * '=')
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            eval_data = {'epoch': int(epoch)}
            eval_data.update(
                eval(args, model, device, 'train', eval_train_loader, epoch))
            eval_data.update(
                eval(args, model, device, 'test', eval_test_loader, epoch))
            eval_df = eval_df.append(pd.Series(eval_data), ignore_index=True)
            logging.info(120 * '=')

        # save stats
        train_df.to_csv(os.path.join(model_dir, 'stats_train.csv'))
        eval_df.to_csv(os.path.join(model_dir, 'stats_eval.csv'))

        # save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            torch.save(dict(num_classes=num_classes,
                            state_dict=model.state_dict(),
                            normalize_input=args.normalize_input),
                       os.path.join(model_dir,
                                    'checkpoint-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir,
                                    'opt-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
