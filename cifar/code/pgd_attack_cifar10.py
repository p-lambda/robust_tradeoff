from __future__ import print_function
import os
import re
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from datasets import SemiSupervisedDataset, DATASETS
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from utils import get_model


def pgd_whitebox(model,
                 X,
                 y,
                 epsilon=8/255,
                 num_steps=20,
                 step_size=0.01):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=8/255,
                  num_steps=20,
                  step_size=0.01):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    logging.info('err pgd black-box: %g', err_pgd.item())
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = pgd_whitebox(model, X, y,
                                               epsilon=args.epsilon,
                                               num_steps=args.num_steps,
                                               step_size=args.step_size)
        logging.info('err pgd (white-box): %g', err_robust.item())
        robust_err_total += err_robust
        natural_err_total += err_natural
    logging.info('natural_err_total: %g', natural_err_total.item())
    logging.info('robust_err_total: %g', robust_err_total.item())


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X,
                                                y,
                                                epsilon=args.epsilon,
                                                num_steps=args.num_steps,
                                                step_size=args.step_size)
        robust_err_total += err_robust
        natural_err_total += err_natural
    logging.info('natural_err_total: %g', natural_err_total.item())
    logging.info('robust_err_total: %g', robust_err_total.item())


def main():

    if args.white_box_attack:
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

        eval_adv_test_whitebox(model, device, test_loader)
    else:
        # black-box attack
        logging.info('pgd black-box attack')
        model = get_model(args.model, num_classes=10)
        if use_cuda:
            model_target = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
        model_target.load_state_dict(torch.load(args.target_model_path))
        model = get_model(args.model, num_classes=10)
        if use_cuda:
            model_source = torch.nn.DataParallel(model).cuda()
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR PGD Attack Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=DATASETS,
                        help='The dataset')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 200)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epsilon', default=0.031, type=float,
                        help='perturbation')
    parser.add_argument('--num-steps', default=20,
                        help='perturb number of steps')
    parser.add_argument('--step-size', default=0.003, type=float,
                        help='perturb step size')
    parser.add_argument('--model-path',
                        default='./checkpoints/model_cifar_wrn.pt',
                        help='model for white-box attack evaluation')
    parser.add_argument('--source-model-path',
                        default='./checkpoints/model_cifar_wrn.pt',
                        help='source model for black-box attack evaluation')
    parser.add_argument('--target-model-path',
                        default='./checkpoints/model_cifar_wrn.pt',
                        help='target model for black-box attack evaluation')
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
    testset = SemiSupervisedDataset(base_dataset=args.dataset,
                                    train=False, root='data',
                                    download=True,
                                    transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.test_batch_size,
                                              shuffle=False, **kwargs)

    main()
