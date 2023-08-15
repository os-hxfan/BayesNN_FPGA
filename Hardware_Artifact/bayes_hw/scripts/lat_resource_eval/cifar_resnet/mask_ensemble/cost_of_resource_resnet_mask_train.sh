# As we just need the model to test hardware performance, number of epoch is set as 5 to reduce the training time
python3 train_qkeras_mcme.py --dataset cifar10 --num_epoch 5 --batch_size 128 --lr 0.01 --mc_samples 3 --gpus 2 --save_model cifar_resnet_tmp_0bayeslayer_mask --quant_tbit 8 --model_name resnet --save_dir ./exp_cifar_bayes_resnet --num_bayes_layer 0 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset cifar10 --num_epoch 5 --batch_size 128 --lr 0.01 --mc_samples 3 --gpus 2 --save_model cifar_resnet_tmp_1bayeslayer_mask --quant_tbit 8 --model_name resnet --save_dir ./exp_cifar_bayes_resnet --num_bayes_layer 1 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset cifar10 --num_epoch 5 --batch_size 128 --lr 0.01 --mc_samples 3 --gpus 2 --save_model cifar_resnet_tmp_2bayeslayer_mask --quant_tbit 8 --model_name resnet --save_dir ./exp_cifar_bayes_resnet --num_bayes_layer 2 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset cifar10 --num_epoch 5 --batch_size 128 --lr 0.01 --mc_samples 3 --gpus 2 --save_model cifar_resnet_tmp_3bayeslayer_mask --quant_tbit 8 --model_name resnet --save_dir ./exp_cifar_bayes_resnet --num_bayes_layer 3 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset cifar10 --num_epoch 5 --batch_size 128 --lr 0.01 --mc_samples 3 --gpus 2 --save_model cifar_resnet_tmp_4bayeslayer_mask --quant_tbit 8 --model_name resnet --save_dir ./exp_cifar_bayes_resnet --num_bayes_layer 4 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset cifar10 --num_epoch 5 --batch_size 128 --lr 0.01 --mc_samples 3 --gpus 2 --save_model cifar_resnet_tmp_5bayeslayer_mask --quant_tbit 8 --model_name resnet --save_dir ./exp_cifar_bayes_resnet --num_bayes_layer 5 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset cifar10 --num_epoch 5 --batch_size 128 --lr 0.01 --mc_samples 3 --gpus 2 --save_model cifar_resnet_tmp_6bayeslayer_mask --quant_tbit 8 --model_name resnet --save_dir ./exp_cifar_bayes_resnet --num_bayes_layer 6 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset cifar10 --num_epoch 5 --batch_size 128 --lr 0.01 --mc_samples 3 --gpus 2 --save_model cifar_resnet_tmp_7bayeslayer_mask --quant_tbit 8 --model_name resnet --save_dir ./exp_cifar_bayes_resnet --num_bayes_layer 7 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset cifar10 --num_epoch 5 --batch_size 128 --lr 0.01 --mc_samples 3 --gpus 2 --save_model cifar_resnet_tmp_8bayeslayer_mask --quant_tbit 8 --model_name resnet --save_dir ./exp_cifar_bayes_resnet --num_bayes_layer 8 --opt_mode temporal --dropout_type mask