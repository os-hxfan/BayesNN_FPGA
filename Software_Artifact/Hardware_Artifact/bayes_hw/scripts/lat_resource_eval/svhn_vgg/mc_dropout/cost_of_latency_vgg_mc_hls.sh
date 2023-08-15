# Require to run the train script under the same directory to run this hls script.
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_spt_2samples_mc --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_spt_2samples_mc_hls --strategy latency  --model_name vgg
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_spt_3samples_mc --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_spt_3samples_mc_hls --strategy latency  --model_name vgg
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_spt_5samples_mc --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_spt_5samples_mc_hls --strategy latency  --model_name vgg
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_spt_7samples_mc --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_spt_7samples_mc_hls --strategy latency  --model_name vgg
python3 hls4ml_build.py --load_model ./exp_svhn_bayes_vgg/svhn_vgg_spt_9samples_mc --output_dir ./exp_svhn_bayes_vgg/svhn_vgg_spt_9samples_mc_hls --strategy latency  --model_name vgg