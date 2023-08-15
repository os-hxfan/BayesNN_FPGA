import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import dict_drop
from train.loss import get_loss_function
from train.loss.loss_functions import ExitEnsembleDistillation

# Various random functions, mostly from: https://github.com/mary-phuong/multiexit-distillation/blob/master/main.py
# or https://github.com/mary-phuong/multiexit-distillation/blob/master/utils.py
def get_device(gpu):
    return torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')

def predict(model, X, gpu):
    device = get_device(gpu)
    X = X.to(device)
    model.to(device)
    scores = model(X)
    # Get last score
    _, pred = scores[-1].max(1)
    return pred.cpu().numpy()

def get_optimizer(model, hyperparameters):
    Opt = getattr(torch.optim, hyperparameters['call'])
    opt = Opt(model.parameters(), **dict_drop(hyperparameters, 'call'))
    return opt

def get_scheduler(opt, hyperparameters):
    Scheduler = getattr(torch.optim.lr_scheduler, hyperparameters['call'])
    scheduler = Scheduler(opt, **dict_drop(hyperparameters, 'call'))
    return scheduler

def validate_model_acc(loss_f, net, val_iter, gpu):
    metrics = []
    with torch.no_grad():
        for val_tuple in val_iter:
            val_tuple = [t.to(get_device(gpu)) for t in val_tuple]
            metrics += [loss_f.metrics(net, *val_tuple)]
    return [sum(metric) / len(metric) for metric in zip(*metrics)]

def validate_model(loss_fn,net,val_iter,gpu, loss_type = "acc"):
    if isinstance(loss_fn, ExitEnsembleDistillation):
        device = get_device(gpu)
        # train is avg acc, val_loss is top_1_acc
        train_acc, val_acc = loss_fn.validate(val_iter, net, device)
        val_loss = 1 - val_acc
        train_loss = 1 - train_acc
        val_loss = val_loss.cpu()
        train_loss = train_loss.cpu()

    elif loss_type == "acc":
        val_metrics = validate_model_acc(loss_fn,net,val_iter,gpu)
        val_loss = 1 - val_metrics[0]
        train_metrics = loss_fn.trn_metrics()
        train_loss = 1 - train_metrics[0]
    elif loss_type == "cross_entropy":
        # What to do about num exits?
        val_hyperparams = dict(       # train with classification loss only
                call = 'ClassificationOnlyLoss',
                n_exits = net.n_exits,
                acc_tops = [1, 5],
            )
        val_loss_fn = get_loss_function(val_hyperparams)
        val_loss = validate_model_ce(val_loss_fn,net,val_iter,gpu)
        val_loss = val_loss.cpu()
    return train_loss, val_loss

def validate_model_ce(loss_fn, net, val_iter, gpu):
    device = get_device(gpu)
    loss = 0
    net.eval()
    for (x,y) in val_iter:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            loss += loss_fn(net,x,y)
    loss /= len(val_iter)
    net.train()
    return loss


def tab_str(*args):
    float_types = (float, torch.FloatTensor, torch.cuda.FloatTensor)
    strings = (f'{a:>8.4f}' if isinstance(a, float_types) else f'{a}'
               for a in args)
    return '\t'.join(strings)

def plot_loss(train_losses, val_losses, experiment_id):
    x_val = range(len(val_losses))
    x = np.linspace(0,len(val_losses), num = len(train_losses))
    plt.scatter(x_val, val_losses, zorder = 10)
    plt.scatter(x,train_losses, zorder = 1)
    plt.savefig("./snapshots/figures/loss_curve_"+str(experiment_id)+".png")