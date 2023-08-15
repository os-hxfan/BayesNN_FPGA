# Require to run the train script under the same directory to run this hls script.
python3 hls4ml_build.py --load_model ./exp_mnist_bayes_lenet/mnist_lenet_spt_2samples_mc --output_dir ./exp_mnist_bayes_lenet/mnist_lenet_spt_2samples_mc_hls --strategy latency  --model_name lenet
python3 hls4ml_build.py --load_model ./exp_mnist_bayes_lenet/mnist_lenet_spt_3samples_mc --output_dir ./exp_mnist_bayes_lenet/mnist_lenet_spt_3samples_mc_hls --strategy latency  --model_name lenet
python3 hls4ml_build.py --load_model ./exp_mnist_bayes_lenet/mnist_lenet_spt_5samples_mc --output_dir ./exp_mnist_bayes_lenet/mnist_lenet_spt_5samples_mc_hls --strategy latency  --model_name lenet
python3 hls4ml_build.py --load_model ./exp_mnist_bayes_lenet/mnist_lenet_spt_7samples_mc --output_dir ./exp_mnist_bayes_lenet/mnist_lenet_spt_7samples_mc_hls --strategy latency  --model_name lenet
python3 hls4ml_build.py --load_model ./exp_mnist_bayes_lenet/mnist_lenet_spt_9samples_mc --output_dir ./exp_mnist_bayes_lenet/mnist_lenet_spt_9samples_mc_hls --strategy latency  --model_name lenet