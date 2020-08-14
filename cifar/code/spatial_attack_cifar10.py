from __future__ import print_function
import os
import itertools
import re
import argparse
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from datasets import SemiSupervisedDataset, DATASETS
from torchvision import transforms
import torch.backends.cudnn as cudnn
from utils import get_model
import spatial
import json

NUM_ROT = 31
NUM_TRANS = 5

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def transform(x, rotation, translation):
    assert x.shape[1] == 3

    with torch.no_grad():
        translated = spatial.transform(x, rotation, translation)

    return translated


def get_spatial_adv_example(model, X, y, max_rot=30, max_trans=0.1071):

    def calc_correct(inp):
        output = model(inp)
        targets = y.repeat([inp.shape[0]])
        return (output.argmax(dim=1) == targets).long()

    with torch.no_grad():
        rots = torch.linspace(-max_rot, max_rot, steps=NUM_ROT)
        trans = torch.linspace(-max_trans, max_trans, steps=NUM_TRANS)
        tfms = torch.tensor(list(itertools.product(rots, trans, trans))).cuda(device=device)
        all_rots = tfms[:, 0]
        all_trans = tfms[:, 1:]

        ntfm = all_rots.shape[0]
        transformed = transform(X.repeat([ntfm, 1, 1, 1]), all_rots, all_trans)
        torch.clamp(transformed, 0, 1.0)

        # X_pgd = Variable(torch.zeros(X.data.shape), requires_grad=True)
        MAX_BS = 128
        i = 0
        while i < ntfm:
            to_do = transformed[i:i+MAX_BS]
            is_correct = calc_correct(to_do)
            argmin = is_correct.argmin()
            if is_correct[argmin] == 0:
                return transformed[i+argmin:i+argmin+1].squeeze_(0)

            i += MAX_BS
        else:
            return transformed[0:1].squeeze_(0)


def apply(func, M):
    tList = [func(m) for m in torch.unbind(M, dim=0)]
    return torch.stack(tList, dim=0)


def get_batch_spatial_adv_example(model, X, y, max_rot=30, max_trans=0.1071, random=False, wo10=False):
    def calc_correct(inp):
        output = model(inp)
        return (output.argmax(dim=1) == y).long()

    if random:
        bs = X.shape[0]
        rots = spatial.unif((bs,), -max_rot, max_rot)
        txs = spatial.unif((bs, 2), -max_trans, max_trans)
        transformed = transform(X, rots, txs)
        return transformed

    elif wo10:
        all_transformed = []
        all_is_corrects = []
        for i in range(10):
            bs = X.shape[0]
            rots = spatial.unif((bs,), -max_rot, max_rot)
            txs = spatial.unif((bs, 2), -max_trans, max_trans)
            transformed = transform(X, rots, txs)
            all_transformed.append(transformed)
            all_is_corrects.append(calc_correct(transformed))
        aic = torch.stack(all_is_corrects, dim=0).argmin(dim=0)
        all_transformed = torch.stack(all_transformed, dim=0)
        X_pgd = []
        for j, i in enumerate(torch.unbind(aic, dim=0)):
            X_pgd.append(all_transformed[i, j])
        X_pgd = torch.stack(X_pgd, dim=0)
        return X_pgd
    else:
        # otherwise grid
        X_pgd = []
        for cur_x, cur_y in zip(torch.unbind(X, dim=0), torch.unbind(y, dim=0)):
            X_pgd.append(get_spatial_adv_example(model, cur_x, cur_y, max_rot, max_trans))
        X_pgd = torch.stack(X_pgd, dim=0)
        return X_pgd


def pgd_whitebox_spatial(model, X, y, max_rot=30, max_trans=0.1071, random=False, eval=False):
    wo10 = (not random and not eval)
    X_pgd = get_batch_spatial_adv_example(model, X, y, max_rot, max_trans, random=random, wo10=wo10)
    err = (model(X).data.max(1)[1] != y.data).float().sum()
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def eval_adv_test_whitebox_spatial(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    total = 0

    for data, target, unsup in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = pgd_whitebox_spatial(model, X, y,
                                               max_rot=args.max_rot,
                                               max_trans=args.max_trans,
                                               eval=True)
        logging.info('err pgd (white-box): %g', err_robust.item())
        robust_err_total += err_robust
        natural_err_total += err_natural
        total += X.shape[0]
    natural_acc = 1.0 - (natural_err_total.item() / total)
    robust_acc = 1.0 - (robust_err_total.item() / total)
    logging.info(f'natural_accuracy: {natural_acc}')
    logging.info(f'robust_accuracy: {robust_acc}')
    stats = {'natural_accuracy': natural_acc, 'robust_accuracy': robust_acc}
    with open(os.path.join(output_dir, 'stats.json'), 'w') as outfile:
        json.dump(stats, outfile)


def main():
    # white-box attack
    logging.info('pgd white-box attack')
    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    num_classes = checkpoint.get('num_classes', 10)
    normalize_input = checkpoint.get('normalize_input', False)
    model = get_model(args.model, num_classes=num_classes,
                      normalize_input=normalize_input)
    if not all([k.startswith('module') for k in state_dict]):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    model.load_state_dict(state_dict)

    eval_adv_test_whitebox_spatial(model, device, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR Spatial Attack Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=DATASETS,
                        help='The dataset')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 200)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--max-rot', default=30, type=int, help='rotation angle')
    parser.add_argument('--max-trans', default=0.1071, type=float, help='translation')
    parser.add_argument('--num-steps', default=20,
                        help='perturb number of steps')
    parser.add_argument('--step-size', default=0.003, type=float,
                        help='perturb step size')
    parser.add_argument('--model-path',
                        default='./checkpoints/model_cifar_wrn.pt',
                        help='model for white-box attack evaluation')
    parser.add_argument('--white-box-attack', default=True,
                        help='whether perform white-box attack')
    parser.add_argument('--model', '-m', default='wrn-34-10', type=str,
                        help='name of the model')
    parser.add_argument('--output-suffix', '-o', default='', type=str,
                        help='string to add to log filename')

    args = parser.parse_args()

    output_dir, checkpoint_name = os.path.split(args.model_path)
    epoch = int(re.search('epoch(\d+)', checkpoint_name).group(1))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir,
                                             'attack_epoch%d%s.log' %
                                             (epoch, args.output_suffix))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    logging.info('PGD attack')
    logging.info('Args: %s', args)

    # settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # set up data loader
    transform_test = transforms.Compose([transforms.ToTensor(), ])
    # testset = torchvision.datasets.CIFAR10(root='data', train=False,
    #                                        download=True,
    #                                        transform=transform_test)
    testset = SemiSupervisedDataset(base_dataset=args.dataset,
                                    train=False, root='data',
                                    download=True,
                                    transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.test_batch_size,
                                              shuffle=False, **kwargs)

    main()

