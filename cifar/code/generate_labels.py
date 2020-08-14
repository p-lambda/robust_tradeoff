from tinyimages import TinyImages

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import logging
import os
import pickle

from utils import get_model, load_cifar10_keywords


import argparse

import numpy as np

from torchvision import transforms

import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import time

import pdb

parser = argparse.ArgumentParser(
    description='Apply clean model to generate labels on unlabeled data')
parser.add_argument('--model_dir', type=str,
                    help='path of checkpoint to standard trained model')
parser.add_argument('--data_dir', default='data/', type=str,
                    help='directory with data')
parser.add_argument('--output_dir', default='data/', type=str,
                    help='directory to save resultant targets')
parser.add_argument('--take_amount_seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--take_fraction', type=float, default=1.0, metavar='S',
                    help='fraction of labeled data (default: 1)')
parser.add_argument('--model', '-m', default='wrn-28-10', type=str,
                    help='name of the model')
parser.add_argument('--model_epoch', '-e', default=200, type=int,
                    help='Number of epochs trained')
parser.add_argument('--output_path', default=None, type=str,
                    help='Output path to force')


args = parser.parse_args()
if not os.path.exists(args.model_dir):
    raise ValueError('Model dir %s not found' % args.model_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.model_dir, 'prediction.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Prediction on unlabeled data')
logging.info('Args: %s', args)


# Loading unlabeled data
logging.info("Loading data")
data_filename = 'ti_top_50000_pred_v3.1.pickle'
with open(os.path.join(args.data_dir, data_filename), 'rb') as f:
        data = pickle.load(f)

logging.info("Loading model")
# Loading model
checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoint-epoch%d.pt' % args.model_epoch))
num_classes = checkpoint.get('num_classes', 10)
normalize_input = checkpoint.get('normalize_input', False)
model = get_model(args.model, 
                  num_classes=num_classes,
                  normalize_input=normalize_input)
model = nn.DataParallel(model).cuda()
model.load_state_dict(checkpoint['state_dict'])

model.eval()

unlabeled_data = CIFAR10(args.data_dir, train=False, transform=ToTensor())
unlabeled_data.data = data['data']
unlabeled_data.targets = list(data['extrapolated_targets'])
data_loader = torch.utils.data.DataLoader(unlabeled_data,
                                          batch_size=256, num_workers=1, pin_memory=True)
predictions = []
for i, (batch, _) in enumerate(data_loader):
    _, preds = torch.max(model(batch.cuda()), dim=1)
    predictions.append(preds.cpu().numpy())

    if (i+1) % 10 == 0:
        print('Done %d/%d' % (i+1, len(data_loader)))


new_extrapolated_targets = np.concatenate(predictions, axis=0)
n = len(new_extrapolated_targets)
print('Difference between new and old extrapolated targets = %.3g%%' %
      (100 * (new_extrapolated_targets != data['extrapolated_targets'][:n]).mean()))


labeled_original_data = CIFAR10(args.data_dir, train=True, transform=ToTensor())
labeled_indices = np.arange(len(labeled_original_data.targets))
rng_state = np.random.get_state()
np.random.seed(args.take_amount_seed)
take_inds = np.random.choice(len(labeled_indices),
                            int(args.take_fraction*len(labeled_indices)), replace=False)
logger = logging.getLogger()
logger.info('Randomly taking only %d/%d examples from training'
            ' set, seed=%d, indices=%s',
            args.take_fraction*len(labeled_indices), len(labeled_indices),
            args.take_amount_seed, take_inds)
np.random.set_state(rng_state)
unlabeled_indices = np.arange(len(labeled_original_data.targets))
new_labeled_indices = labeled_indices[take_inds]
new_unlabeled_indices = list(set(unlabeled_indices) - set(new_labeled_indices))

labeled_original_data.data = labeled_original_data.data[new_unlabeled_indices]
labeled_original_data.targets = [labeled_original_data.targets[i] for i in new_unlabeled_indices]

data_loader = torch.utils.data.DataLoader(labeled_original_data,
                                          batch_size=256, num_workers=1, pin_memory=True)

for i, (batch, _) in enumerate(data_loader):
    _, preds = torch.max(model(batch.cuda()), dim=1)
    predictions.append(preds.cpu().numpy())

    if (i+1) % 10 == 0:
        print('Done %d/%d' % (i+1, len(data_loader)))


new_extrapolated_targets = np.concatenate(predictions, axis=0)

new_targets = dict(extrapolated_targets=new_extrapolated_targets,
                   prediction_model=args.model_dir,
                   prediction_model_epoch=args.model_epoch)

if args.output_path is None:
    out_filename ='targets-model='+str(args.model)+'-take_frac=' + str(args.take_fraction) + '-take_seed=' + str(args.take_amount_seed) + '.pickle'
    out_path = os.path.join(args.output_dir, out_filename)
else:
    out_path = args.output_path

#assert(not os.path.exists(out_path))
with open(out_path, 'wb') as f:
        pickle.dump(new_targets, f)
