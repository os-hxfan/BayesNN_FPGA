cd ../software/
#Resnet18

# Mask

python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.125 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 --mask_type mask --mask_scale 4 2>&1 > resnet18_cifar100_mask_scale4.txt

python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.125 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 --mask_type mask --mask_scale 5 2>&1 > resnet18_cifar100_mask_scale5.txt

python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.125 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 --mask_type mask --mask_scale 6 2>&1 > resnet18_cifar100_mask_scale6.txt

# Mask+ME

python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 0 --n_epochs 200 --dropout_exit True --dropout_p 0.125 --reducelr_on_plateau True --dataset_name cifar100 --mask_type mask --mask_scale 4 2>&1 > resnet18_cifar100_maskme_scale4.txt

python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 0 --n_epochs 200 --dropout_exit True --dropout_p 0.125 --reducelr_on_plateau True --dataset_name cifar100 --mask_type mask --mask_scale 5 2>&1 > resnet18_cifar100_maskme_scale5.txt

python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 0 --n_epochs 200 --dropout_exit True --dropout_p 0.125 --reducelr_on_plateau True --dataset_name cifar100 --mask_type mask --mask_scale 6 2>&1 > resnet18_cifar100_maskme_scale6.txt



# VGG19

# Mask

python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.25 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 --mask_type mask --mask_scale 3 2>&1 > vgg19_cifar100_mask_scale3.txt

python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.25 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 --mask_type mask --mask_scale 4 2>&1 > vgg19_cifar100_mask_scale4.txt

python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.25 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 --mask_type mask --mask_scale 5 2>&1 > vgg19_cifar100_mask_scale5.txt

python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.25 --reducelr_on_plateau True --single_exit True --dataset_name cifar100 --mask_type mask --mask_scale 6 2>&1 > vgg19_cifar100_mask_scale6.txt


# Mask+ME

python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.375 --reducelr_on_plateau True --dataset_name cifar100 --mask_type mask --mask_scale 3 2>&1 > vgg19_cifar100_maskme_scale3.txt

python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.375 --reducelr_on_plateau True --dataset_name cifar100 --mask_type mask --mask_scale 4 2>&1 > vgg19_cifar100_maskme_scale4.txt

python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.375 --reducelr_on_plateau True --dataset_name cifar100 --mask_type mask --mask_scale 5 2>&1 > vgg19_cifar100_maskme_scale5.txt

python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.375 --reducelr_on_plateau True --dataset_name cifar100 --mask_type mask --mask_scale 6 2>&1 > vgg19_cifar100_maskme_scale6.txt


