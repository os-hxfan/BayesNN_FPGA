#!/usr/bin/env python3
import sys
sys.path.append(sys.path[0] + '/..')
sys.path.append(sys.path[0] + '/../converter/keras')
sys.path.append(sys.path[0] + '/models')

from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import *
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.utils import to_categorical
from qkeras import *
from tensorflow.keras.optimizers import Adam, SGD
import os
import argparse 
import numpy as np
from models import lenet, ResNet18, VGG11
from converter.keras.MCDropout import BayesianDropout
from converter.keras.Masksembles import MasksemblesModel, Masksembles
from qmodels import Qlenet, QResNet18, QVGG11, QVIBNN
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from data_utils import CIFAR10Data
from scipy.io import loadmat
from svhn.utils import CosineAnnealingScheduler
from model_utils import Top_Level_Model

model_num_layer = {"lenet": 3, "resnet": 3}

def get_dataset(args):
    if args.dataset == "mnist":
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        RESHAPED = 784

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

        x_train /= 256.0
        x_test /= 256.0
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
    elif args.dataset == "cifar10":
        cifar10_data = CIFAR10Data()
        x_train, y_train, x_test, y_test = cifar10_data.get_data(subtract_mean=True)

        num_train = int(x_train.shape[0] * 0.9)
        num_val = x_train.shape[0] - num_train
        mask = list(range(num_train, num_train+num_val))
        x_val = x_train[mask]
        y_val = y_train[mask]

        mask = list(range(num_train))
        x_train = x_train[mask]
        y_train = y_train[mask]

        print('num train:%d num val:%d' % (num_train, num_val))
        data = (x_train, y_train, x_val, y_val, x_test, y_test)
    elif args.dataset == "svhn":
        # Pre-processing, get from https://www.kaggle.com/code/dimitriosroussis/svhn-classification-with-cnn-keras-96-acc
        np.random.seed(20)
        train_raw = loadmat('./svhn/train_32x32.mat')
        test_raw = loadmat('./svhn/test_32x32.mat')
        train_images = np.array(train_raw['X'])
        test_images = np.array(test_raw['X'])

        train_labels = train_raw['y']
        test_labels = test_raw['y']
        train_images = np.moveaxis(train_images, -1, 0)
        test_images = np.moveaxis(test_images, -1, 0)
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        train_labels = train_labels.astype('int32')
        test_labels = test_labels.astype('int32')
        train_images /= 255.0
        test_images /= 255.0
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        train_labels = lb.fit_transform(train_labels)
        test_labels = lb.fit_transform(test_labels)
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels,
                                                        test_size=0.15, random_state=22)
    else:
        raise NotImplementedError("Dataset not supoorted")
    
    if args.dataset == "cifar10":
        dataset = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test, "x_val": x_val, "y_val": y_val} 
    else:
        dataset = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}
    return dataset

def get_model(args):
    #model=Sequential()
    if args.model_name == "lenet":
        if args.is_quant != 0:
            model = Qlenet(args, model_num_layer[args.model_name])
        else:
            model = lenet(args, model_num_layer[args.model_name])
    elif args.model_name == "vibnn": # vibnn is deprecated as we found its permance is very bad in HLS4ML
        model = QVIBNN(args)
    elif args.model_name == "resnet":
        if args.is_quant != 0:
            model = QResNet18(input_shape=(32, 32, 3), classes=10, args=args, weight_decay=1e-4, base_filters=16)
        else:
            model = ResNet18(input_shape=(32, 32, 3), classes=10, args=args, weight_decay=1e-4, base_filters=16)
    elif args.model_name == "vgg":
        if args.is_quant != 0:
            model = QVGG11(args, filters=16, dense_out=[16, 16, 10])
        else:
            model = VGG11(args, filters=16, dense_out=[16, 16, 10])
    else:
        raise NotImplementedError("Model not supoorted") 
    #model.compile(optimizer=SGD(lr = args.lr), loss=['categorical_crossentropy'], metrics=['accuracy'])
    print(model.summary())
    return model


def train(args, model, dataset): 

    if args.model_name == "lenet" or args.model_name == "vibnn": # vibnn is deprecated as we found its permance is very bad in HLS4ML
        train_stat = model.fit(
            dataset['x_train'], dataset['y_train'], batch_size=args.batch_size,
            epochs=args.num_epoch, initial_epoch=1,
            validation_split=args.validation_split)
    elif args.model_name == "resnet":
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=4,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=4,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
        )
        print('train with data augmentation')
        train_gen = datagen.flow(dataset['x_train'], dataset['y_train'], batch_size=args.batch_size)
        # def lr_scheduler(epoch):
        #     lr = args.lr
        #     new_lr = lr
        #     if epoch <= 91:
        #         pass
        #     elif epoch > 91 and epoch <= 137:
        #         new_lr = lr * 0.1
        #     else:
        #         new_lr = lr * 0.01
        #     print('new lr:%.2e' % new_lr)
        #     return new_lr 
        def lr_scheduler(epoch):
            lr = args.lr
            new_lr = lr * (0.1 ** (epoch // 50))
            print('new lr:%.2e' % new_lr)
            return new_lr 
        reduce_lr = CosineAnnealingScheduler(T_max=args.num_epoch, eta_max=args.lr, eta_min=1e-4)
        history = model.fit_generator(generator=train_gen,
                                           epochs=args.num_epoch,
                                           callbacks=[reduce_lr],
                                           validation_data=(dataset['x_val'], dataset['y_val']),
                                           )
    elif args.model_name == "vgg":
        callbacks = [CosineAnnealingScheduler(T_max=args.num_epoch, eta_max=args.lr, eta_min=1e-4)]
        datagen = ImageDataGenerator(rotation_range=8,
                                    zoom_range=[0.95, 1.05],
                                    height_shift_range=0.10,
                                    shear_range=0.15)
        train_stat = model.fit(datagen.flow(dataset['x_train'], dataset['y_train'], batch_size=args.batch_size),
            epochs=args.num_epoch, validation_data=(dataset['x_test'], dataset['y_test']), callbacks=callbacks)
    else:
        raise NotImplementedError("Training not supoorted") 

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mnist", type=str, required=True, help="Name of dataset")
    parser.add_argument("--save_dir", default="./exp_mnist_bayes_lenet", type=str, required=True, help="Directory name of saved results")
    parser.add_argument("--model_name", default="lenet", type=str, help="Name of contructed model")
    
    parser.add_argument("--gpus", default="0,1", type=str, required=True, help="GPUs id, separated by comma without space, e.g, 0,1,2")
    parser.add_argument("--save_model", default=None, type=str, help="Name of save model")
    parser.add_argument("--is_train", default=1, type=int, help="Whether to train model")
    parser.add_argument("--is_quant", default=1, type=int, help="Whether to quantize model")
    parser.add_argument("--load_model", default=None, type=str, help="Name of load model")

    parser.add_argument("--validation_split", default=0.1, type=float, help="Validation slipt of dataset")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--num_epoch", default=100, type=int, required=True, help="The number of epoch for training")
    parser.add_argument("--num_bayes_layer", default=0, type=int, help="The number of Bayesian Layer")
    parser.add_argument("--quant_tbit", default=6, type=int, help="The total bits of quant")
    parser.add_argument("--quant_ibit", default=0, type=int, help="The integer bits of quant")
    parser.add_argument("--dropout_rate", default=0.2, type=float, help="The dropout rate")
    parser.add_argument("--num_masks", default=4, type=int, help="The number of masks")
    parser.add_argument("--scale", default=4, type=float, help="The scale")
    parser.add_argument("--mc_samples", default=5, type=int, help="The number of MC samples")
    
    parser.add_argument("--batch_size", default=64, type=int, help="The number of batches for training")
    parser.add_argument("--dropout_type", default="mc", type=str, choices=["mc", "mask"], help="Dropout type, Monte-Carlo Dropout (mc) or Mask Ensumble (mask)")
    parser.add_argument("--is_me", default=0, type=int, help="Whether use multi-exit, 0 denote no use")
    parser.add_argument("--num_exits", default=2, type=int, help="The number of exits in multi-exit arch")

    args = parser.parse_args()

    # Set GPU environment
    gpus = args.gpus.split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)

    if not os.path.exists(args.save_dir):
        print ("Create Non-exiting Directory")
        os.makedirs(args.save_dir)
    args.save_model = args.save_dir + "/" + args.save_model
    # Get dataset and model
    dataset = get_dataset(args)
    if args.load_model is None:
        model = get_model(args)            
    else:
        load_model_name = args.load_model + '.h5'
        co = {"BayesianDropout": BayesianDropout, "Masksembles": Masksembles, "MasksemblesModel": MasksemblesModel}
        _add_supported_quantized_objects(co)
        model = load_model(load_model_name, custom_objects=co)
    model = Top_Level_Model(args, model)    

    if args.is_train != 0:
        train(args, model, dataset)
    scores = model.evaluate(dataset["x_test"], dataset["y_test"], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    if args.save_model is not None: 
        model.model.save(args.save_model+'.h5')

    #eval(args, model, dataset)
