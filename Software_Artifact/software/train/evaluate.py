import torch
import numpy as np
from utils import dict_drop
from train.train_utils import validate_model_acc, tab_str, get_device

# Evaluation function is inspired from https://github.com/mary-phuong/multiexit-distillation/blob/master/evaluate.py
# Includes heavy modifications to fit the more modular structure and MC dropout
def evaluate(loss_fn, test_iter, model, gpu, experiment_id, mc_dropout_passes, create_log = True):
    # Setting train mode to false
    model.eval()
    loss_metrics = loss_fn.metric_names
    test_metrics = np.zeros((mc_dropout_passes,len(loss_metrics)))
    for i in range(mc_dropout_passes):
        values = validate_model_acc(loss_fn, model, test_iter, gpu)
        values[-1] = values[-1].item()
        test_metrics[i] = values
    averaged_test_metrics = list(np.average(test_metrics, axis = 0))
    std_test_metrics = list(np.std(test_metrics, axis = 0))
    # Log
    if create_log:
        log(loss_metrics,averaged_test_metrics, experiment_id)
    return averaged_test_metrics

def log(loss_metrics,val_metrics, experiment_id):
    float_types = (float, torch.FloatTensor, torch.cuda.FloatTensor)
    value_list = [f'{a:>8.4f}' if isinstance(a, float_types) else f'{a}'
               for a in val_metrics]
    joined = list(zip(loss_metrics,value_list))
    with open("log_"+str(experiment_id)+".txt", "w") as file1:
        # Writing data to a file
        file1.write(str(joined))