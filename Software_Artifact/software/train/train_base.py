import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train.train_utils import get_device, validate_model, tab_str, plot_loss

# Structure is from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop
# Slight modifications for grad clipping and grad accumulation
def train_single_epoch(model, data_loader, optimizer, loss_fn, device, dtype = torch.float32, max_norm = 10, grad_accumulation = None):
    running_loss = 0
    last_loss = 0
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    optimizer.zero_grad()
    for i, (x,y) in enumerate(data_loader):
        # Every data instance is an input + label pair
        x = x.to(device=device, dtype=dtype)  # move to device
        y = y.to(device=device, dtype=torch.long)

        # Compute the loss and its gradients
        loss = loss_fn(model,x, y)
        loss.backward()

        # Needed for training
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Adjust learning weights
        if grad_accumulation is not None:
            if i % grad_accumulation or i == (len(data_loader) - 1):
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

        # Gather data and report
        running_loss += loss.item()
        
        if i % 200 == 199:
            last_loss = running_loss / 200 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss

# Structure is primarily from the 2022 Deep Learning CW1 task:
# https://gitlab.doc.ic.ac.uk/lab2122_spring/DL_CW_1_lrc121/-/blob/master/dl_cw_1.ipynb
# Modifications are made to include additional arguments and early stopping
def train_loop(model, optimizer, scheduler,  data_loaders, loss_fn, experiment_id, max_norm = 1, patience = 20, epochs=1, 
                gpu = -1, val_loss_type = "acc", grad_accumulation = None):
                            # Change to str as opposed to bool
    """
    Train a model
    """
    device = get_device(gpu)
    train_loader, val_loader = data_loaders
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    # Sets to train mode
    model.train()
    best_val_loss = float("inf")
    counter = 0
    all_train_losses = []
    all_val_losses = []
    for e in range(epochs):
        train_loss, val_loss = validate_model(loss_fn, model, val_loader, gpu, loss_type = val_loss_type)
        last_loss = train_single_epoch(model,train_loader,optimizer,loss_fn, device, max_norm = max_norm, grad_accumulation=grad_accumulation)
        train_loss, val_loss = validate_model(loss_fn, model, val_loader, gpu, loss_type = val_loss_type)
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        print(f"epoch: {e}, actual loss: {tab_str(last_loss)}, train_loss: {tab_str(train_loss)}, val_loss: {tab_str(val_loss)}, learning_rate: {optimizer.param_groups[0]['lr']}")
        # had issues with trn_metrics, remove
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model, "./snapshots/best_val_model_"+str(experiment_id))
        else:
            counter += 1
            if counter > patience:
                break
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
    plot_loss(all_train_losses,all_val_losses, experiment_id)
    return model

