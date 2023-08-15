import sacred
import argparse
import torch
import models
import train
import datasets
from utils import RUNS_DB_DIR

# Inspired by https://github.com/mary-phuong/multiexit-distillation/blob/master/train.py
sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
sacred.SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = False
ex = sacred.Experiment()
ex.observers.append(sacred.observers.FileStorageObserver.create(str(RUNS_DB_DIR)))

parser = argparse.ArgumentParser(description="Adding dropout")
parser.add_argument('--dropout_exit', type=bool, default=False)
parser.add_argument('--dropout_p', type=float, default=0.5)
parser.add_argument('--dropout_type', type=str, default=None)
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=300)
parser.add_argument('--full_analysis_and_save', type=bool, default = False)
parser.add_argument('--single_exit', type=bool, default=False)
parser.add_argument('--backbone', type=str, default = "msdnet")
parser.add_argument('--grad_clipping', type=float, default = 2)
parser.add_argument('--gpu', type=int,default=0)
parser.add_argument('--val_split', type=float, default = 0.1)
parser.add_argument('--reducelr_on_plateau', type=bool, default = False)
parser.add_argument('--dataset_name', type=str, default = 'cifar100')
parser.add_argument('--grad_accumulation',type=int,default = 0)
parser.add_argument("--mask_type", default="mc", type=str, choices=["mc", "mask"], help="Dropout type, Monte-Carlo Dropout (mc) or Mask Ensumble (mask)")
parser.add_argument('--num_masks', type=int, default = 4)
parser.add_argument('--mask_scale', type=float, default = 4)

args = parser.parse_args()
#https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true

# Specify Hyperparameters (maybe add command line compatibility?)
hyperparameters = train.get_hyperparameters(args)
# Load any Utilities (like a timer, logging...)
ex.add_config(hyperparameters)

@ex.main
def main(_config):
    # Get experiment ID for logging purposes
    experiment_id = ex.current_run._id
    # Load the dataset
    print("Loading Datasets")
    # Convert hyperparameters so they load into this directly (and the above)!
    # Need to return validation stuff as well!
    train_loader, val_loader, test_loader = datasets.get_dataloader(hyperparameters["loaders"])

    # Load the Network
    print("Creating Network")
    model = models.get_network(hyperparameters["network"])
    # Print Network Architec
    print (model)
    # Train the Network
    print("Starting Training")
    # Get loss function (if class need to initialize it then run it)
    loss_fn = train.get_loss_function(hyperparameters["loss"])
    # Get Optimizer
    optimizer = train.get_optimizer(model,hyperparameters["optimizer"])
    # Get Scheduler
    scheduler = train.get_scheduler(optimizer,hyperparameters["scheduler"])
    # Train Loop (do i need to return model?)
    model = train.train_loop(model,optimizer,scheduler,(train_loader,val_loader),
            loss_fn, experiment_id, gpu = hyperparameters["gpu"], 
            epochs = hyperparameters["n_epochs"], 
            patience = hyperparameters["patience"],
            max_norm = hyperparameters["max_norm"],
            val_loss_type = hyperparameters["val_loss_type"],
            grad_accumulation = hyperparameters["grad_accumulation"]) # model, optimizer, scheduler,  data_loaders, loss_fn, epochs=1, gpu = -1

    # Evaluate the Network on Test
    test_loss_fn = train.get_loss_function(hyperparameters["test_loss"])
    # loss_fn, test_iter, model, gpu
    results = train.evaluate(test_loss_fn, test_loader,model,hyperparameters["gpu"], experiment_id, hyperparameters["mc_dropout_passes"])
    # Save Model
    torch.save(model, "./snapshots/final_model_"+str(experiment_id))
    # Define suffix
    suffix = ""
    if args.single_exit is False: suffix += "me_"
    if args.dropout_exit is True: 
        suffix += args.mask_type
        if args.mask_type == "mask":
            suffix += "_scale" + str(int(args.mask_scale))
        else:
            suffix += "_droprate" + str(int(args.dropout_p))

    # Define analyzer
    if args.full_analysis_and_save:
        dropout = False
        if args.dropout_exit or args.dropout_type is not None:
            dropout = True
        full_analyzer = train.FullAnalysis(model, test_loader, gpu = hyperparameters["gpu"], 
            mc_dropout = dropout, mc_passes = hyperparameters["mc_dropout_passes"], suffix = suffix)
        full_analyzer.all_experiments(experiment_id)
        full_analyzer.save_validation(experiment_id, val_loader)
        full_analyzer.get_confidence_exiting_values(experiment_id)
    return results

if __name__ == "__main__":
    ex.run()