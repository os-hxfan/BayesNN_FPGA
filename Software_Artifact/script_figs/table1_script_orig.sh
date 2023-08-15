cd ../software/
#Resnet18

# Baseline
python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 2 --n_epochs 200 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 2>&1 > resnet18_cifar100_baseline.txt

# MCD

# Acc-Opt
python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.125 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 2>&1 > resnet18_cifar100_mc_acc_opt.txt

# ECE-Opt
python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.5 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 2>&1 > resnet18_cifar100_mc_ece_opt.txt

# ME

# Acc-Opt (0.9 Confidence Threshold, although in paper is 0.95)
python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 0 --n_epochs 200 --reducelr_on_plateau True --dataset_name cifar100 2>&1 > resnet18_cifar100_me_acc_opt.txt

# ECE-Opt (0.95 Confidence Threshold)
python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 0 --n_epochs 200 --reducelr_on_plateau True --dataset_name cifar100 2>&1 > resnet18_cifar100_me_ece_opt.txt


# MCD+ME

# Acc-Opt (Ensemble 3)
python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 0 --n_epochs 200 --dropout_exit True --dropout_p 0.125 --reducelr_on_plateau True --dataset_name cifar100 2>&1 > resnet18_cifar100_mcme_acc_opt.txt

# ECE-Opt (0.5 Confidence Threshold)
python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 0 --n_epochs 200 --dropout_exit True --dropout_p 0.25 --reducelr_on_plateau True --dataset_name cifar100 2>&1 > resnet18_cifar100_mcme_ece_opt.txt


# VGG19

# Baseline
python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 2>&1 > vgg19_cifar100_baseline.txt

# MCD

# Acc-Opt
python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.25 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 2>&1 > vgg19_cifar100_mc_acc_opt.txt

# ECE-Opt
python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.25 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 2>&1 > vgg19_cifar100_mc_ece_opt.txt

# ME

# Acc-Opt (Ensemble 3)
python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --reducelr_on_plateau True --dataset_name cifar100 2>&1 > vgg19_cifar100_me_acc_opt.txt


# Acc-Opt (0.5 Confidence Threshold)
python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --reducelr_on_plateau True --dataset_name cifar100 2>&1 > vgg19_cifar100_me_ece_opt.txt


# MCD+ME

# Acc-Opt (Ensemble 3)
python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.375 --reducelr_on_plateau True --dataset_name cifar100 2>&1 > vgg19_cifar100_mcme_acc_opt.txt


# ECE-Opt (0.5 Confidence Threshold)
python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.125 --reducelr_on_plateau True --dataset_name cifar100 2>&1 > vgg19_cifar100_mcme_ece_opt.txt












