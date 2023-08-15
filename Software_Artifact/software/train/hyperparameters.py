
# The dictionary set up of the various hyperparameters is from: https://github.com/mary-phuong/multiexit-distillation/blob/master/train_cifar.py
def get_hyperparameters(args):
    # Main
    model_type = args.backbone
    n_epochs = args.n_epochs
    gpu = args.gpu
    patience = args.patience
    
    # Train and Val Loaders
    loader_hyperparameters, dataset_image_size, dataset_out_dim = get_loader_hyperparameters(args)
    # Network
    network_hyperparameters, mc_dropout_passes = get_network_hyperparameters(model_type, args, dataset_image_size, dataset_out_dim)
    # Losses
    loss_hyperparameters, val_loss_type = get_loss_hyperparameters(network_hyperparameters["n_exits"], model_type, args)
    test_loss_hyperparameters = get_test_hyperparameters(network_hyperparameters["n_exits"], model_type, args)
    # Optimizer and Scheduler
    opt_hyperparameters, sched_hyperparameters, max_norm, grad_accumulation = get_opt_sched_hyperparameters(args)
    
    
    hyperparameters = dict(
        network = network_hyperparameters,
        loss = loss_hyperparameters,
        optimizer = opt_hyperparameters,
        scheduler = sched_hyperparameters,
        n_epochs = n_epochs,
        test_loss = test_loss_hyperparameters,
        gpu = gpu,
        patience = patience,
        mc_dropout_passes = mc_dropout_passes,
        max_norm = max_norm,
        val_loss_type = val_loss_type,
        loaders = loader_hyperparameters,
        grad_accumulation = grad_accumulation,
        )
    return hyperparameters

def get_network_hyperparameters(model_type, args, dataset_image_size, dataset_out_dim):
    if model_type == "msdnet":
        hyperparams = dict(          # MSDNet architecture parameters
            call = 'MsdNet',
            in_shape = 32,
            out_dim = 100,
            n_scales = 3,
            n_exits = 11,
            nlayers_to_exit = 4,
            nlayers_between_exits = 2,
            nplanes_mulv = [6, 12, 24],
            nplanes_addh = 1,
            nplanes_init = 1,
            prune = 'min',
            plane_reduction = 0.5, # Try this with 0 to avoid the halving
            exit_width = 128, # same as 128 dim 3x3 filters in exit?
            dropout = args.dropout_type,
            dropout_exit = args.dropout_exit,
            dropout_p = args.dropout_p,
            load_model = None,
            )
        if args.single_exit:
            hyperparams["execute_exits"] = [hyperparams["n_exits"] - 1]
    elif model_type == "resnet20":
        hyperparams = dict(
            call = "ResNet20",
            resnet_type = "early_exit",
            load_model = None,
            out_dim = 100,
            exit_after = -1, # exit after all blocks
            complexity_factor = 1.2, #effects size of intermediary layers? Set to default
            dropout = args.dropout_type,
            dropout_exit = args.dropout_exit,
            dropout_p = args.dropout_p,
            n_exits = 10 # Doesn't affect network, but does effect loss!!!
        )
        if args.single_exit and (args.dropout_exit or args.dropout_type is not None):
            hyperparams["resnet_type"] = "mc"
            hyperparams["n_exits"] = 1            

        elif args.single_exit:
            hyperparams["resnet_type"] = None
            hyperparams["n_exits"] = 1
        
        elif args.dropout_exit or args.dropout_type is not None:
            hyperparams["resnet_type"] = "mc_early_exit"

    elif model_type == "resnet18" or model_type == "vgg19":
        hyperparams = dict(
            call = "ResNet18",
            resnet_type = "early_exit",
            load_model = None,
            out_dim = 100,
            dropout = args.dropout_type,
            dropout_exit = args.dropout_exit,
            dropout_p = args.dropout_p,
            n_exits = 4 # Doesn't affect network, but does effect loss!!!
        )
        if model_type == "vgg19":
            hyperparams["call"] = "VGG19"
            hyperparams["n_exits"] = 5
            
        if args.single_exit and (args.dropout_exit or args.dropout_type is not None):
            hyperparams["resnet_type"] = "mc"
            hyperparams["n_exits"] = 1            

        elif args.single_exit:
            hyperparams["resnet_type"] = None
            hyperparams["n_exits"] = 1
        
        elif args.dropout_exit or args.dropout_type is not None:
            hyperparams["resnet_type"] = "mc_early_exit"

    if hyperparams["dropout"] is not None or hyperparams["dropout_exit"]:
        mc_dropout_passes = 10
    else:
        mc_dropout_passes = 1
    hyperparams["out_dim"] = dataset_out_dim
    hyperparams["image_size"] = dataset_image_size
    hyperparams["mask_type"] = args.mask_type
    hyperparams["num_masks"] = args.num_masks
    hyperparams["mask_scale"] = args.mask_scale
    return hyperparams, mc_dropout_passes

def get_loss_hyperparameters(num_exits, model_type, args, loss_type = "distillation_annealing"):
    if model_type == "msdnet":
        if loss_type == "distillation_annealing" and not args.single_exit:
            loss = dict(         # distillation-based training with temperature
                                # annealing
            call = 'DistillationBasedLoss',
            n_exits = num_exits,
            acc_tops = [1, 5],
            
            C = 0.5, # Confidence Limit (?)
            maxprob = 0.5, 
            global_scale = 2.0 * 5/num_exits, # Not mentioned in paper
            # Temperature multiplier is 1.05 by default
            )
        elif loss_type == "distillation_constant" and not args.single_exit:
            loss = dict(       # distillation-based training with constant
                                # temperature
                call = 'DistillationLossConstTemp',
                n_exits = num_exits,
                acc_tops = [1, 5],
                C = 0.5,
                T = 4.0,
                global_scale = 2.0 * 5/num_exits,
            )
        elif loss_type == "classification" or args.single_exit:
            loss = dict(       # train with classification loss only
                call = 'ClassificationOnlyLoss',
                n_exits = num_exits,
                acc_tops = [1, 5],
            )
    elif model_type == "resnet18" or model_type == "vgg19":
        loss = dict(
            call = "ExitEnsembleDistillation",
            n_exits = num_exits,
            acc_tops = [1,5],
            use_EED = True, 
            loss_output = "MSE", 
            use_feature_dist = False, # beta = 0
            temperature = 3 # default value from paper
        )
        
    elif model_type == "resnet20":
        # Add standard loss function stuff here
        if loss_type == "distillation_annealing" and not args.single_exit:
            loss = dict(         # distillation-based training with temperature
                                # annealing
            call = 'DistillationBasedLoss',
            n_exits = num_exits,
            acc_tops = [1, 5],
            
            C = 0.5, # Confidence Limit (?)
            maxprob = 0.5, 
            global_scale = 2.0 * 5/num_exits, # Not mentioned in paper
            # Temperature multiplier is 1.05 by default
            )
        elif loss_type == "classification" or args.single_exit:
            loss = dict(       # train with classification loss only
                call = 'ClassificationOnlyLoss',
                n_exits = num_exits,
                acc_tops = [1, 5],
            )
    val_loss_type = "acc"
    return loss, val_loss_type

def get_opt_sched_hyperparameters(args):
    cf_opt = dict(          # optimization method
    call = 'SGD',
    lr = 0.5, # Note this is from Paper 9 (Paper 10 used 0.1)
    momentum = 0.9,
    weight_decay = 1e-4,
    nesterov = True,
    )
    cf_scheduler = dict(   # learning rate schedule
    call = "ReduceLROnPlateau",
    factor = 0.5,
    patience = 10,
    # call = 'MultiStepLR',
    # milestones = [150, 225],
    #gamma = 0.1
    )
    
    if args.backbone == "vgg19":
        cf_opt = dict(          # optimization method
        call = 'SGD',
        lr = 0.1, # Note this is from Paper 9 (Paper 10 used 0.1)
        momentum = 0.9,
        weight_decay = 5e-4,
        )  
        if args.reducelr_on_plateau:
            cf_scheduler = dict(   # learning rate schedule
            call = "ReduceLROnPlateau",
            factor = 0.1,
            patience = 10,)
        else:
            cf_scheduler = dict(   # learning rate schedule
            call = "CosineAnnealingLR",
            T_max = 200,)

    if args.backbone == "resnet18":
        cf_opt = dict(          # optimization method
        call = 'SGD',
        lr = 0.1, # Note this is from Paper 9 (Paper 10 used 0.1)
        momentum = 0.9,
        weight_decay = 5e-4,
        nesterov = True,
        )  
        if args.reducelr_on_plateau:
            cf_scheduler = dict(   # learning rate schedule
            call = "ReduceLROnPlateau",
            factor = 0.1,
            patience = 10,)
        else:
            cf_scheduler = dict(   # learning rate schedule
            call = 'MultiStepLR',
            milestones = [75,130,180],
            gamma = 0.1)

    if args.dataset_name == "chestx":
        cf_opt = dict(          # optimization method
            call = 'Adam',
            lr = 0.0005,
            weight_decay = 0 
            )  
        cf_scheduler = dict(   # learning rate schedule
            call = "ReduceLROnPlateau",
            factor = 0.1,
            patience = 5,)
            
    max_norm = args.grad_clipping
    if args.grad_clipping == 0:
        max_norm = None
    grad_accumulation = args.grad_accumulation
    if args.grad_accumulation == 0:
        grad_accumulation = None
    return cf_opt, cf_scheduler, max_norm, grad_accumulation

def get_loader_hyperparameters(args):
    hyperparameters = dict(dataset_name = args.dataset_name,
        batch_size = (64,64,250), #(train, val, test) 
        # train and val batch sizes should be the same for plotting purposes
        augment = True,
        val_split = args.val_split,
        )
    if args.backbone == "resnet18" or args.backbone == "resnet20" or args.backbone == "vgg19":
        hyperparameters["batch_size"] = (128,128,250)
        
    if hyperparameters["dataset_name"] == "chestx":
        size = 224
        out_dim = 7
        hyperparameters["batch_size"] = (16,16,32)
    else:
        size = 32
        out_dim = 100
    return hyperparameters, size, out_dim


def get_test_hyperparameters(n_exits, model_type,args):
    cf_loss = dict(  # evaluation metric
        call = 'MultiExitAccuracy',
        n_exits = n_exits,
        acc_tops = (1,5),
    )
    if args.single_exit:
        cf_loss["n_exits"] = 1
    return cf_loss
