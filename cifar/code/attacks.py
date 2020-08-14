import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
import logging
import foolbox.foolbox
from foolbox.foolbox.models import PyTorchModel
from foolbox.foolbox.attacks import CarliniWagnerLIAttack
from foolbox.foolbox.distances import Linf

def cw(model,
       X,
       y,
       binary_search_steps=5,
       max_iterations=1000,
       learning_rate=5E-3,
       initial_const=1E-2,
       tau_decrease_factor=0.9
       ):

    foolbox_model = PyTorchModel(model, bounds=(0, 1), num_classes=10)
    attack = CarliniWagnerLIAttack(foolbox_model,distance=Linf)
    linf_distances= []
    for i in range(len(X)):
        logging.info('Example: %g', i)
        image = X[i, :].detach().cpu().numpy()
        label = y[i].cpu().numpy()
        adversarial = attack(image, label,
                             binary_search_steps=binary_search_steps,
                             max_iterations=max_iterations,
                             learning_rate=learning_rate,
                             initial_const=initial_const,
                             tau_decrease_factor=tau_decrease_factor)
        logging.info('Linf distance: %g', np.max(np.abs(adversarial-image)))
        linf_distances.append(np.max(np.abs(adversarial-image)))
    return linf_distances


def pgd(model,
        X,
        y,
        epsilon=8/255,
        num_steps=20,
        step_size=0.01,
        pretrain=False, 
        random_start=True):

    if pretrain:
        out=model(2*X - 1)
    else:
        out=model(X)
    indices_natural = (out.data.max(1)[1] == y.data).float().cpu()
    perturbation = torch.zeros_like(X, requires_grad=True)
    if random_start:
        perturbation = torch.rand_like(X, requires_grad=True)
        perturbation.data = perturbation.data*2*epsilon - epsilon
    matrix_cols=[]
    for _ in range(num_steps):
        # This is just to compute gradient
        opt = optim.SGD([perturbation], lr=1e-3)
        opt.zero_grad()
        
        with torch.enable_grad():
            if pretrain:
                loss = nn.CrossEntropyLoss()(model(2*(X + perturbation)-1), y)
            else:
                loss = nn.CrossEntropyLoss()(model(X + perturbation), y)
        loss.backward()

        perturbation.data = (perturbation + step_size*perturbation.grad.detach().sign()).clamp(-epsilon, epsilon)
        perturbation.data = torch.min(torch.max(perturbation.detach(), -X), 1-X) # clip X+delta to [0,1]
        X_pgd = Variable(torch.clamp(X.data + perturbation.data, 0, 1.0), requires_grad=False)
        if pretrain:
            matrix_cols.append(np.reshape((model(2*(X_pgd)-1).data.max(1)[1] == y.data).float().cpu(), [-1, 1]))
        else:
            matrix_cols.append(np.reshape((model(X_pgd).data.max(1)[1] == y.data).float().cpu(), [-1, 1]))
    matrix=np.concatenate(matrix_cols, axis=1)
    return indices_natural, matrix

