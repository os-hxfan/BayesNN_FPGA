# As we just need the model to test hardware performance, number of epoch is set as 10 to reduce the training time
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --mc_samples 3 --gpus 0 --save_model svhn_vgg_tmp_0bayeslayer_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --num_bayes_layer 0 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --mc_samples 3 --gpus 0 --save_model svhn_vgg_tmp_1bayeslayer_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --num_bayes_layer 1 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --mc_samples 3 --gpus 0 --save_model svhn_vgg_tmp_2bayeslayer_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --num_bayes_layer 2 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --mc_samples 3 --gpus 0 --save_model svhn_vgg_tmp_3bayeslayer_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --num_bayes_layer 3 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --mc_samples 3 --gpus 0 --save_model svhn_vgg_tmp_4bayeslayer_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --num_bayes_layer 4 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --mc_samples 3 --gpus 0 --save_model svhn_vgg_tmp_5bayeslayer_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --num_bayes_layer 5 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --mc_samples 3 --gpus 0 --save_model svhn_vgg_tmp_6bayeslayer_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --num_bayes_layer 6 --opt_mode temporal --dropout_type mask
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --mc_samples 3 --gpus 0 --save_model svhn_vgg_tmp_7bayeslayer_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --num_bayes_layer 7 --opt_mode temporal --dropout_type mask