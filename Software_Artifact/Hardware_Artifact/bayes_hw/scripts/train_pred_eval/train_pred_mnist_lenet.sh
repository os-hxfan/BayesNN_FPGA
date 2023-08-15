python3 train_qkeras.py --dataset mnist --num_epoch 100 --batch_size 128 --lr 0.01 --gpus 0,1,2,3 --save_dir ./exp_mnist_bayes_lenet --save_model mnist_bayes_lenet_3samples_mc --quant_tbit 8 --mc_samples 3 --model_name lenet --dropout_type mc
python3 train_qkeras.py --dataset mnist --num_epoch 100 --batch_size 128 --lr 0.01 --gpus 0,1,2,3 --save_dir ./exp_mnist_bayes_lenet --save_model mnist_bayes_lenet_3samples_mask --quant_tbit 8 --mc_samples 3 --model_name lenet --dropout_type mask
python3 hls4ml_pred.py --load_model ./exp_mnist_bayes_lenet/mnist_bayes_lenet_3samples_mc --dataset mnist --gpus 0,1,2,3 --dropout_type mc --mc_samples 3
python3 hls4ml_pred.py --load_model ./exp_mnist_bayes_lenet/mnist_bayes_lenet_3samples_mask --dataset mnist --gpus 0,1,2,3 --dropout_type mask --mc_samples 3