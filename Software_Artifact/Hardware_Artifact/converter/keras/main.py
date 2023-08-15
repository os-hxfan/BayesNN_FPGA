import argparse
import sys 
from train import mnist_train, cifar_train
from test.LeNet import LeNet
from test.ResNet import ResNet50

parser = argparse.ArgumentParser(description='Run the training model with or without Bayesian conversion')
parser.add_argument('--model', type=str,
                  default='LeNet', help='which model do you want to use? Available models are: LeNet, ResNet50.')
parser.add_argument('--dataset', type=str, default='mnist', help='which dataset do you want to use? Available datasets are: mnist, cifar10')
parser.add_argument('--train', type=bool, default=True, help='whether the model will be trained')
parser.add_argument('--save', type=str, default='model', help='saved model name')
parser.add_argument('--p', type=float, default=0.25,
                    help='dropout probability')
parser.add_argument('--samples', type=int, default=-1, help = 'number of samples')

args = parser.parse_args()
if args.model == 'LeNet':
  model = LeNet()
elif args.model == 'ResNet50':
  model = ResNet50()
else:
  print('{} is not available' % args.model)
  sys.exit(1)

if args.dataset == 'mnist':
  train_fn = mnist_train
elif args.dataset == 'cifar10':
  train_fn = cifar_train
else: 
  print('{} is not available' % args.dataset)
  sys.exit(1)

train_fn(model, args.train, args.save != None, args.save, True, 10)
