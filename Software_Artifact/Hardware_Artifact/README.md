# Hardware Implementation

This folder contains two artifacts:
- AutoBayes that converts a CNN to dropout-based BNN (including MCD and Masksemble). 
- FPGA-based accelerator for dropout-based Baysian Neural Network, based on hls4ml and qkeras

## 1. Structure

```
.
├── README.md
├── autobayes            # All the scripts to generate data in the paper
    ├── models           # Keras models used for the experiment and testing
    ├── timing           # Timing reports generated here
    ├── diff_dropouts    # Synthesis reports for BNN with different number of dropout layers generated here
├── converter            # All code related to converting NN to BNN
    ├── keras            # Code for converting keras NN model to BNN model
    ├── pytorch          # Code for converting torch NN model to BNN model
├── bayes_hw                 # FPGA-based hardware accelerator for BayesNNs, based on qkears and hls4ml 
    ├── models               # Keras model definition
    ├── scripts              # Scripts to generate the tables and figures in the paper
└── requirements.txt        
```

## 2. Limitation

### 2.1 Some known issues of dependencies
- Issue of qkeras on large model: Train is very instable. Sometimes different random seeds will make the training crash.
- Issue of hls4ml: The prediction of hls4ml on large model is weird. For example, the same scripts that work on Lenet will lead to accuracy problem in VGG and ResNet (the accuracy under qkeras has been valided. So the problem should be hls4ml).
- Better automation between pytorch and qkeras.

### 2.2 Future work
- Integrate lastest version of qkeras and hls4ml in our future version.
- More automatic integration between pytorch and qkeras part, and other optimization process of our framework.

## 3. Environment Setup

### 3.1 Dependencies Install

We can use conda to manage the environments.
```
conda create -n autobayes python=3.9
conda activate autobayes
pip3 install -r requirements.txt
```
If you are using Linux machine, make sure you set the environment variables for vivado
```
export LC_ALL=C; unset LANGUAGE
source /PATH/TO/Vivado/2019.2/settings64.sh
```

### 3.2 Dataset Download

Download `SVHN` dataset: 
```
cd ./bayes_hw/svhn
bash download_svhn.sh
```
The `**.mat` files will be downloaded under the same directory.

## 4. Artifact Evaluation

### 4.1 AutoBayes

This part converts a PyTorch or Keras neural network model to a Bayesian neural network model using Morte-Carlo Dropout.

All the scripts are put under the folder `./autobayes`, we refer reviewers/users to read [autobayes](./experiment/README.md) to run it.

### 4.2 FPGA-based Accelerator

The synthesis and place&route may take several days to weeks depending on your machines. You can quickly check our reports in this [link](https://drive.google.com/drive/folders/1ldXGsGuJGxp8IPaYSD3CTrNEn71reTME?usp=sharing).

#### 4.2.1 Model Train and Prediction
The scripts are placed under `./bayes_hw/scripts/train_pred_eval/`. To run Lenet Experiment, follow:
```
cd bayes_hw
bash scripts/train_pred_eval/train_pred_mnist_lenet_mcme.sh
```
#### 4.2.1 Latency and Resource Evaluating
The scripts are placed under `./bayes_hw/scripts/lat_resource_eval/`. To run a LeNet example, using:
```
cd bayes_hw
python3 train_qkeras_mcme.py --dataset mnist --num_epoch 5 --batch_size 128 --lr 0.01 --gpus 1 --save_model mnist_lenet_spt_3samples_mask --quant_tbit 8 --model_name lenet --save_dir ./exp_mnist_bayes_lenet --opt_mode spatial --mc_samples 3 --dropout_type mask
python3 hls4ml_build.py --load_model ./exp_mnist_bayes_lenet/mnist_lenet_spt_3samples_mask --output_dir ./exp_mnist_bayes_lenet/mnist_lenet_spt_3samples_mask_hls --strategy latency  --model_name lenet
```
All other Lenet experiments can be easily run in the following scripts:
```
bash scripts/lat_resource_eval/mnist_lenet/mask_ensemble/cost_of_latency_lenet_mask.sh  # Use Latency strategy to get results.
bash scripts/lat_resource_eval/mnist_lenet/mask_ensemble/cost_of_resource_lenet_mask.sh  # Use Resource strategy to get results.
```