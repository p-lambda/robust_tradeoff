from __future__ import print_function
import os
import json
import numpy as np
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
from attacks import pgd, cw
import torch.backends.cudnn as cudnn
from utils import get_model

def eval_adv_test(model, device, test_loader, attack, attack_params, results_dir):
    """
    evaluate model by white-box attack
    """
    model.eval()
    count = 0
    if attack == 'pgd':
        restarts_matrices = []
        for run in range(attack_params['num_restarts']):
            current_matrix_rows=[]
            count = 0
            batch_num=0
            natural_accuracy = 0
            for data, target , unsup in test_loader:
                batch_num=batch_num+1
                if batch_num > args.num_eval_batches:
                    break 
                data, target = data.to(device), target.to(device)
                count = count + len(target)
                X, y = Variable(data, requires_grad=True), Variable(target)
                # matrix_robust has batch_size*num_iterations dimensions
                indices_natural, matrix_robust = pgd(model, X, y,
                                                     epsilon=attack_params['epsilon'],
                                                     num_steps=attack_params['num_steps'],
                                                     step_size=attack_params['step_size'],
                                                     pretrain=args.pretrain, 
                                                     random_start=attack_params['random_start'])
                natural_accuracy = natural_accuracy + sum(indices_natural)
                current_matrix_rows.append(matrix_robust)
            logging.info('Completed restart: %g' % run)
            current_matrix=np.concatenate(current_matrix_rows, axis=0)
            logging.info("Shape of current_matrix %s" % str(current_matrix.shape))
            restarts_matrices.append(current_matrix)

            final_matrix=np.asarray(restarts_matrices)
            if run == 0:
                final_success=np.ones([count, 1])
            final_success=np.multiply(final_success, np.reshape(current_matrix[:, attack_params['num_steps']-1], [-1, 1]))
            logging.info('%d' % np.sum(final_success))
            # logging.info('%d' % (np.sum(final_success))/count)
            logging.info("Shape of final matrix: %s" % str(final_matrix.shape))
            # logging.info("Final accuracy %g:" % np.sum(final_success)/count)
            stats = {'attack':'pgd',
                     'count': count, 
                     'epsilon': attack_params['epsilon'],
                     'num_steps': attack_params['num_steps'],
                     'step_size': attack_params['step_size'],
                     'natural_accuracy': float(natural_accuracy.item()/count),
                     'final_matrix': final_matrix.tolist(),
                     'robust_accuracy': float(np.sum(final_success)/count),
                     'restart_num':run
            }

            json_stats = {'attack': 'pgd',
                          'natural_accuracy': float(natural_accuracy.item()/count),
                          'robust_accuracy': float(np.sum(final_success)/count)}
            with open(os.path.join(results_dir, 'stats.json'), 'w') as outfile:
                json.dump(json_stats, outfile)

            np.save(os.path.join(results_dir, 'stats' + str(run)+'.npy'), stats)
            np.save(os.path.join(results_dir, 'stats.npy'), stats)
    elif attack == 'cw':
        all_linf_distances = []
        count = 0 
        for data, target, unsup in test_loader:
            logging.info('Batch: %g', count)
            count = count+1
            if count > args.num_eval_batches:
                break
            data, target = data.to(device), target.to(device)
            X, y = Variable(data, requires_grad=True), Variable(target)
            batch_linf_distances = cw(model, X, y,
                                      binary_search_steps=attack_params['binary_search_steps'],
                                      max_iterations=attack_params['max_iterations'],
                                      learning_rate=attack_params['learning_rate'],
                                      initial_const=attack_params['initial_const'],
                                      tau_decrease_factor=attack_params['tau_decrease_factor'])
            all_linf_distances.append(batch_linf_distances)
            np.savetxt(os.path.join(results_dir, 'CW_dist'), all_linf_distances)
        np.savetxt(os.path.join(results_dir, 'CW_dist'), all_linf_distances)
    else:
        raise ValueError("Attack not supported")
        
def main():

    # white-box attack
    logging.info('pgd white-box attack')
    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    num_classes = checkpoint.get('num_classes', 10)
    normalize_input = checkpoint.get('normalize_input', False)
    model = get_model(args.model, num_classes=num_classes,
                      normalize_input=normalize_input,
                      pretrain=args.pretrain)
    if not all([k.startswith('module') for k in state_dict]):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    model.load_state_dict(state_dict)

    attack_params={
        'epsilon':args.epsilon,
        'num_restarts':args.num_restarts,
        'step_size':args.step_size,
        'num_steps':args.num_steps,
        'random_start':args.random_start,
        'binary_search_steps':args.binary_search_steps,
        'max_iterations':args.max_iterations,
        'learning_rate':args.learning_rate,
        'initial_const':args.initial_const,
        'tau_decrease_factor':args.tau_decrease_factor,
        'seed':args.random_seed
    }
           
    print(attack_params)
    eval_adv_test(model, device, test_loader, attack=args.attack,
                  attack_params=attack_params, results_dir=results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR Attack Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=DATASETS,
                        help='The dataset')
    parser.add_argument('--test_batch_size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 200)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epsilon', default=0.031, type=float,
                        help='perturbation')
    parser.add_argument('--attack', default='pgd', type=str,
                        help='attack type: one of (pgd, momentum, cw)')
    parser.add_argument('--num_steps', default=20, type=int, 
                        help='perturb number of steps')
    parser.add_argument('--step_size', default=0.003, type=float,
                        help='perturb step size')
    parser.add_argument('--model_path',
                        default='./checkpoints/model_cifar_wrn.pt',
                        help='model for white-box attack evaluation')
    parser.add_argument('--model', '-m', default='wrn-34-10', type=str,
                        help='name of the model')
    parser.add_argument('--output_suffix', default='', type=str,
                        help='string to add to log filename')
    parser.add_argument('--no_random_start', dest='random_start', action='store_false')
    parser.add_argument('--num_restarts', default=1, type=int,
                        help='Number of restarts for running attack')
    parser.add_argument('--binary_search_steps', default=5, type=int,
                        help='Number of binary search steps for cw attack')
    parser.add_argument('--max_iterations', default=1000, type=int,
                        help='Max number of adam iterations in each CW optimization')
    parser.add_argument('--learning_rate', default=5E-3, type=float,
                        help='Learning rate for CW attack')
    parser.add_argument('--initial_const', default=1E-2, type=float,
                        help='Initial constant for CW attacks')
    parser.add_argument('--tau_decrease_factor', default=0.9, type=float,
                        help='Tau decrease factor (CW)')
    parser.add_argument('--random_seed', default=0, type=int,
                        help='Random seed for permutation of test instances')
    parser.add_argument('--num_eval_batches', default=50, type=int,
                        help='Number of batches to run evalaution on')
    parser.add_argument('--pretrain', action='store_true', default=False)
    

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)

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

    results_dir = os.path.join(output_dir, args.output_suffix)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    
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
    
    if (args.attack == 'cw') or (args.num_eval_batches<100) :
        np.random.seed(123)
        print("Permuted testset")
        permutation = np.random.permutation(len(testset))
        testset.data = testset.data[permutation, :]
        testset.targets = [testset.targets[i] for i in permutation]

        
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.test_batch_size,
                                              shuffle=False, **kwargs)

    main()
