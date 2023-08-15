import train.loss.loss_functions as loss
from utils import dict_drop

# Inspired from how load classes: https://github.com/mary-phuong/multiexit-distillation/blob/master/main.py 
def get_loss_function(hyperparameters):
    if hyperparameters['call'] == 'ExitEnsembleDistillation' or hyperparameters['call'] == 'MultiExitAccuracy':
        loss_f = getattr(loss, hyperparameters['call'])(**dict_drop(hyperparameters, 'call'))
    else:
        raise ValueError("This is not a valid type of loss function.")
    return loss_f

