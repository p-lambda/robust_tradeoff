from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import Sampler, Dataset
from torchvision import transforms
import torch
import numpy as np

import os
import pickle
import pdb

import logging

from hashlib import md5

DATASETS = ['cifar10', 'svhn']


class SemiSupervisedDataset(Dataset):
    def __init__(self,
                 base_dataset='cifar10',
                 downsample=1,
                 take_fraction=None,
                 take_amount_seed=1,
                 semisupervised=False,
                 sup_labels=None,
                 unsup_labels=None,
                 test_labels=None,
                 add_cifar100=False,
                 add_svhn_extra=False,
                 aux_data_filename=None,
                 aux_targets_filename=None,
                 add_aux_labels=False,
                 aux_take_amount=None,
                 aux_label_noise=None,
                 train=False,
                 **kwargs):

        if base_dataset == 'cifar10':
            self.dataset = CIFAR10(train=train, **kwargs)
        elif base_dataset == 'svhn':
            if train:
                self.dataset = SVHN(split='train', **kwargs)
            else:
                self.dataset = SVHN(split='test', **kwargs)
            # because torchvision is annoying
            self.dataset.targets = self.dataset.labels
            self.targets = list(self.targets)

            if train and add_svhn_extra:
                svhn_extra = SVHN(split='extra', **kwargs)
                self.data = np.concatenate([self.data, svhn_extra.data])
                self.targets.extend(svhn_extra.labels)
        else:
            raise ValueError('Dataset %s not supported' % base_dataset)
        self.base_dataset = base_dataset
        self.train = train
        self.transform = self.dataset.transform

        if self.train:
            # Collecting subset of train data with relevant labels
            if sup_labels is not None:
                self.sup_indices = [i for (i, label) in enumerate(self.targets)
                                    if label in sup_labels]
            else:
                self.sup_indices = np.arange(len(self.targets))
                sup_labels = range(max(self.targets) + 1)

            # Collecting subset of train data with relevant labels
            if unsup_labels is not None:
                self.unsup_indices = [i for (i, label) in
                                      enumerate(self.targets)
                                      if label in unsup_labels]
            else:
                self.unsup_indices = np.arange(len(self.targets))

            self.sup_indices = self.sup_indices[::downsample]
            if take_fraction is not None:
                rng_state = np.random.get_state()
                np.random.seed(take_amount_seed)
                take_inds = np.random.choice(len(self.sup_indices),
                                             int(take_fraction*len(self.sup_indices)),
                                             replace=False)
                np.random.set_state(rng_state)

                logger = logging.getLogger()
                logger.info('Randomly taking only %d/%d examples from training'
                            ' set, seed=%d, indices=%s',
                            take_fraction*len(self.sup_indices), len(self.sup_indices),
                            take_amount_seed, take_inds)
                self.sup_indices = self.sup_indices[take_inds]

            self.unsup_indices = list(set(self.unsup_indices)
                                      - set(self.sup_indices))

            if semisupervised:
                labeled = [self.targets[i] for i in self.sup_indices]
                labeled = [sup_labels.index(i) for i in labeled]
                unlabeled = [-1] * len(self.unsup_indices)
                self.targets = labeled + unlabeled
                self.data = np.concatenate((self.data[self.sup_indices],
                                            self.data[self.unsup_indices]),
                                           axis=0)
                self.sup_indices = list(range(len(self.sup_indices)))
                self.unsup_indices = list(
                    range(len(self.sup_indices),
                          len(self.sup_indices)+len(self.unsup_indices)))
                # self.train_labels = [
                #     label if i % downsample == 0 else -1
                #     for (i, label) in enumerate(self.train_labels)]
            else:
                self.all_targets = np.copy(self.targets)
                self.all_data = np.copy(self.data)
                self.targets = [self.targets[i] for i in self.sup_indices]
                self.targets = [sup_labels.index(i) for i in self.targets]
                self.data = self.data[self.sup_indices, ...]
                self.sup_indices = list(range(len(self.sup_indices)))

            self.orig_len = len(self.data)
            if add_cifar100:
                orig_len = len(self.data)
                cifar100 = CIFAR100(**kwargs)
                self.data = np.concatenate((self.data, cifar100.data), axis=0)
                self.targets.extend([-1] * len(cifar100.targets))
                self.unsup_indices.extend(
                    range(orig_len, orig_len + len(cifar100)))

            if aux_data_filename is not None:
                aux_path = os.path.join(kwargs['root'], aux_data_filename)
                print("Loading data from %s" % aux_path)
                with open(aux_path, 'rb') as f:
                    aux = pickle.load(f)
                aux_data = aux['data']
                aux_targets = aux['extrapolated_targets']
                orig_len = len(self.data)

                if aux_take_amount is not None:
                    rng_state = np.random.get_state()
                    np.random.seed(take_amount_seed)
                    take_inds = np.random.choice(len(aux_data),
                                                 aux_take_amount, replace=False)
                    np.random.set_state(rng_state)

                    logger = logging.getLogger()
                    logger.info(
                        'Randomly taking only %d/%d examples from aux data'
                        ' set, seed=%d, indices=%s',
                        aux_take_amount, len(aux_data),
                        take_amount_seed, take_inds)
                    aux_data = aux_data[take_inds]
                    aux_targets = aux_targets[take_inds]

                if not add_aux_labels:
                    self.targets.extend([-1] * len(aux_data))
                else:
                    if aux_targets_filename is not None:
                        aux_path = aux_targets_filename
                        print("Loading data from %s" % aux_path)
                        with open(aux_path, 'rb') as f:
                            aux = pickle.load(f)
                        new_aux_targets = aux['extrapolated_targets']
                        n = len(aux_targets)
                        print('Difference between new and old extrapolated targets = %.3g%%' %
                              (100 * (aux_targets != new_aux_targets[:n]).mean()))

                        if (len(new_aux_targets) > len(aux_targets)):
                            assert(len(new_aux_targets) - len(aux_targets) == len(self.unsup_indices))
                            true_labels = [self.all_targets[i] for i in self.unsup_indices]
                            print('Difference between extrapolated and true labels on training set = %.3g%%' %
                                  (100 * (true_labels != new_aux_targets[n:]).mean()))
                            logging.info('Adding unsupervised %d examples from training data' %(len(self.unsup_indices)))
                            unlabeled_data = self.all_data[self.unsup_indices]
                            # Since new targets are now included
                            self.unsup_indices = []
                            aux_data = np.concatenate((aux_data, unlabeled_data), axis=0)
                        aux_targets = new_aux_targets

                    else:
                        self.unsup_indices=[]
                    if aux_label_noise:
                        num_aux = len(aux_targets)
                        num_to_noise = int(num_aux * aux_label_noise)
                        logging.info('Making %d/%d aux labels noisy, '
                                     'numpy rng state MD5=%s' %
                                     (num_to_noise, num_aux,
                                      md5(np.random.get_state()[1]).hexdigest()
                                      ))
                        inds_to_noise = np.random.choice(
                            num_aux, num_to_noise, replace=False)
                        permutated_labels = np.random.permutation(
                            aux_targets[inds_to_noise])
                        aux_targets[inds_to_noise] = permutated_labels

                    self.targets.extend(aux_targets)
                self.data = np.concatenate((self.data, aux_data), axis=0)
                # note that we use unsup indices to track the labeled datapoints
                # whose labels are "fake"
                self.unsup_indices.extend(
                    range(orig_len, orig_len+len(aux_data)))

                self.orig_len = orig_len
            logger = logging.getLogger()
            logger.info("Training set")
            logger.info("Number of training samples: %d", len(self.targets))
            logger.info("Number of supervised samples: %d",
                        len(self.sup_indices))
            logger.info("Number of unsup samples: %d", len(self.unsup_indices))
            logger.info("Label histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of training data: %s", np.shape(self.data))

        # Test set
        else:
            self.orig_len = len(self.data)
            if test_labels is not None:
                self.test_indices = [i for (i, label) in enumerate(self.targets)
                                     if label in test_labels]
                self.targets = [self.targets[i] for i in self.test_indices]
                self.targets = [test_labels.index(i) for i in self.targets]
                self.data = self.data[self.test_indices, ...]
            self.orig_len = len(self.data)
            logger = logging.getLogger()
            logger.info("Test set")
            logger.info("Number of samples: %d", len(self.targets))
            logger.info("Label histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of data: %s", np.shape(self.data))

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets  # because torchvision is annoying
        d = self.dataset[item]
        d = list(d)
        d.append(item >= self.orig_len)
        d = tuple(d)
        return d
        # return self.dataset[item]


    def __repr__(self):
        fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SemiSupervisedSampler(Sampler):
    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5,
                 num_batches=None):
        if unsup_fraction is None or unsup_fraction < 0:
            self.sup_inds = sup_inds + unsup_inds
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

        self.batch_size = batch_size
        unsup_batch_size = int(batch_size * unsup_fraction)
        self.sup_batch_size = batch_size - unsup_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(
                np.ceil(len(self.sup_inds) / self.sup_batch_size))


        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            sup_inds_shuffled = [self.sup_inds[i]
                                 for i in torch.randperm(len(self.sup_inds))]
            for sup_k in range(0, len(self.sup_inds), self.sup_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = sup_inds_shuffled[sup_k:(sup_k + self.sup_batch_size)]
                # extending with unlabeled data
                if self.sup_batch_size < self.batch_size:
                    batch.extend([self.unsup_inds[i] for i in
                                  torch.randint(high=len(self.unsup_inds),
                                                size=(
                                                    self.batch_size - len(
                                                        batch),),
                                                dtype=torch.int64)])

                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches
