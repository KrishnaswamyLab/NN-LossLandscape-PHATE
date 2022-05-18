'''
Script adapted from https://github.com/pluskid/fitting-random-labels
'''

from __future__ import print_function

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim
import random

from torchvision.datasets import CIFAR10, CIFAR100
from cifar10_data import CIFAR10RandomLabels
from cifar10_data import CIFAR100RandomLabels
from cifar10_data import MNISTRandomLabels
from cifar10_data import FashionMNISTRandomLabels

from torch.utils.data.sampler import SubsetRandomSampler

import cmd_args
import model_mlp, model_wideresnet

import net_plotter_seed

def get_data_loaders(args, shuffle_train=True):
  if args.data == 'cifar10' and args.retrain_label_corrupt_prob == 0:
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.retrain_data_augmentation:
      transform_train_valid = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
          ])
    else:
      transform_train_valid = transforms.Compose([
          transforms.ToTensor(),
          normalize,
          ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train_valid)
    valid_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train_valid)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.retrain_valid_size * num_train))

    if shuffle_train:
        np.random.seed(4)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.retrain_batch_size, sampler=train_sampler, **kwargs,
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.retrain_batch_size, sampler=valid_sampler, **kwargs,
        )

    test_loader = torch.utils.data.DataLoader(
        CIFAR10(root='./data', train=False, download=True, transform=transform_test),
        batch_size=args.retrain_batch_size, shuffle=False, **kwargs,
        )

    return train_loader, valid_loader, test_loader
  elif args.data == 'cifar10' and args.retrain_label_corrupt_prob > 0:
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.retrain_data_augmentation:
      transform_train_valid = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
          ])
    else:
      transform_train_valid = transforms.Compose([
          transforms.ToTensor(),
          normalize,
          ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    train_dataset = CIFAR10RandomLabels(root='./data', train=True, download=True,
                        transform=transform_train_valid, num_classes=args.num_classes,
                        corrupt_prob=args.retrain_label_corrupt_prob)
    valid_dataset = CIFAR10RandomLabels(root='./data', train=True, download=True,
                        transform=transform_train_valid, num_classes=args.num_classes,
                        corrupt_prob=args.retrain_label_corrupt_prob)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.retrain_valid_size * num_train))

    if shuffle_train:
        np.random.seed(4)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.retrain_batch_size, sampler=train_sampler, **kwargs,
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.retrain_batch_size, sampler=valid_sampler, **kwargs,
        )

    test_loader = torch.utils.data.DataLoader(
        CIFAR10RandomLabels(root='./data', train=False,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.retrain_label_corrupt_prob),
        batch_size=args.retrain_batch_size, shuffle=False, **kwargs,
        )

    return train_loader, valid_loader, test_loader
  elif args.data == 'cifar100' and args.retrain_label_corrupt_prob == 0:
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.retrain_data_augmentation:
      transform_train_valid = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
          ])
    else:
      transform_train_valid = transforms.Compose([
          transforms.ToTensor(),
          normalize,
          ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_train_valid)
    valid_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_train_valid)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.retrain_valid_size * num_train))

    if shuffle_train:
        np.random.seed(4)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.retrain_batch_size, sampler=train_sampler, **kwargs,
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.retrain_batch_size, sampler=valid_sampler, **kwargs,
        )

    test_loader = torch.utils.data.DataLoader(
        CIFAR100(root='./data', train=False, download=True, transform=transform_test),
        batch_size=args.retrain_batch_size, shuffle=False, **kwargs,
        )
    return train_loader, valid_loader, test_loader
  else:
    raise Exception('Unsupported dataset: {0}'.format(args.data))


def get_model(args):
  # create model
  random.seed(args.retrain_rand_seed)
  np.random.seed(args.retrain_rand_seed)
  torch.manual_seed(args.retrain_rand_seed)
  torch.cuda.manual_seed_all(args.retrain_rand_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  if args.arch == 'wide-resnet':
    if args.data == 'cifar10' or args.data == 'cifar100':
      model = model_wideresnet.WideResNet(args.wrn_depth, args.num_classes,
                                        args.wrn_widen_factor,
                                        drop_rate=args.wrn_droprate)
    elif args.data == 'mnist' or args.data=='fashion_mnist':
      model = model_wideresnet_mnist.WideResNet(args.wrn_depth, args.num_classes,
                                        args.wrn_widen_factor,
                                        drop_rate=args.wrn_droprate)
  elif args.arch == 'mlp':
    n_units = [int(x) for x in args.mlp_spec.split('x')] # hidden dims
    n_units.append(args.num_classes)  # output dim
    if args.data == 'cifar10' or args.data == 'cifar100':
      n_units.insert(0, 32*32*3)
    elif args.data == 'mnist' or args.data=='fashion_mnist':
      n_units.insert(0, 28*28*1)        # input dim
    model = model_mlp.MLP(n_units)

  # for training on multiple GPUs.
  # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
  # model = torch.nn.DataParallel(model).cuda()
  model = model.cuda()

  return model


def retrain_model(args, model, train_loader, valid_loader, test_loader,
                start_epoch=None, epochs=None):
  #cudnn.benchmark = True
  random.seed(args.retrain_rand_seed)
  np.random.seed(args.retrain_rand_seed)
  torch.manual_seed(args.retrain_rand_seed)
  torch.cuda.manual_seed_all(args.retrain_rand_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # define loss function (criterion) and pptimizer
  criterion = nn.CrossEntropyLoss().cuda()
  optimizer = torch.optim.SGD(model.parameters(), args.retrain_learning_rate,
                              momentum=args.retrain_momentum,
                              weight_decay=args.retrain_weight_decay)

  start_epoch = start_epoch or 1
  epochs = epochs or (args.retrain_epochs + 1)
  exp_dir = os.path.join('runs', args.exp_name, args.retrain_name)

  #--------------------------------------------------------------------------
  train_loss, train_acc = validate_epoch(train_loader, model, criterion, 0, args)
  valid_loss, valid_acc = validate_epoch(valid_loader, model, criterion, 0, args)
  test_loss, test_acc = validate_epoch(test_loader, model, criterion, 0, args)

  state = {
      'train_accuracy': train_acc,
      'train_loss': train_loss,
      'valid_accuracy': valid_acc,
      'valid_loss': valid_loss,
      'test_accuracy': test_acc,
      'test_loss': test_loss,
      'epoch': 0,
      'state_dict': model.state_dict(keep_vars=True)
  }
  opt_state = {
      'optimizer': optimizer.state_dict()
  }
  torch.save(state, exp_dir + '/model_e0.t7')
  torch.save(opt_state, exp_dir + '/opt_state_e0.t7')
  logging.info('%03d: Acc-tr: %6.2f, Acc-val: %6.2f, Acc-test: %6.2f, L-tr: %6.4f, L-val: %6.4f, L-test: %6.4f',
               0, train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss)
  #--------------------------------------------------------------------------

  for epoch in range(start_epoch, epochs):
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, args)

    # evaluate on validation and test datasets
    valid_loss, valid_acc = validate_epoch(valid_loader, model, criterion, epoch, args)
    test_loss, test_acc = validate_epoch(test_loader, model, criterion, 0, args)

    if args.eval_full_trainset:
      train_loss, train_acc = validate_epoch(train_loader, model, criterion, epoch, args)

    #--------------------------------------------------------------------------
    if epoch == args.retrain_epochs or epoch % args.retrain_save_epoch == 0:
    #if epoch % args.save_epoch == 0 or epoch == 150:
        state = {
            'train_accuracy': train_acc,
            'train_loss': train_loss,
            'valid_accuracy': valid_acc,
            'valid_loss': valid_loss,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'epoch': epoch,
            'state_dict': model.state_dict(keep_vars=True),
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, exp_dir + '/model_e' + str(epoch) + '_end.t7')
        torch.save(opt_state, exp_dir + '/opt_state_e' + str(epoch) + '_end.t7')
    #--------------------------------------------------------------------------

    logging.info('%03d: Acc-tr: %6.2f, Acc-val: %6.2f, Acc-test: %6.2f, L-tr: %6.4f, L-val: %6.4f, L-test: %6.4f',
                 epoch, train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss)


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
  """Train for one epoch on the training set"""
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to train mode
  model.train()

  if args.retrain_saves_per_epoch > 1:
      num_batches = len(train_loader)
      save_ev_x_batches = np.ceil(num_batches/args.retrain_saves_per_epoch)
      exp_dir = os.path.join('runs', args.exp_name, args.retrain_name)

  for i, (input, target) in enumerate(train_loader):
    target = target.cuda(non_blocking=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1 = accuracy(output.data, target, topk=(1,))[0]
    losses.update(loss.item(), input.size(0))
    top1.update(prec1.item(), input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.retrain_saves_per_epoch > 1 and i != 0 and i%save_ev_x_batches == 0:
        state = {
            'epoch': epoch,
            'batch': i,
            'state_dict': model.state_dict(keep_vars=True),
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, exp_dir + '/model_e' + str(epoch) + '_' + str(i // save_ev_x_batches)+ '.t7')
        torch.save(opt_state, exp_dir + '/opt_state_e' + str(epoch) + '_' + str(i // save_ev_x_batches)+ '.t7')

  return losses.avg, top1.avg


def validate_epoch(valid_loader, model, criterion, epoch, args):
  """Perform validation on the validation set
     Basically a forward pass with accuracy/loss evaluation"""

  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(valid_loader):
    target = target.cuda(non_blocking=True)
    input = input.cuda()
    with torch.no_grad():
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

  return losses.avg, top1.avg


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
  ''' Not used during the retraining.'''
  # """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
  # lr = args.retrain_learning_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
  # for param_group in optimizer.param_groups:
  #     param_group['lr'] = lr
  pass


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res


def retrain_setup_logging(args):
  import datetime
  exp_dir = os.path.join('runs', args.exp_name, args.retrain_name)
  if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)
  log_fn = os.path.join(exp_dir, "LOG.{0}.txt".format(datetime.date.today().strftime("%y%m%d")))
  logging.basicConfig(filename=log_fn, filemode='w', level=logging.DEBUG)
  # also log into console
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)
  print('Logging into %s...' % exp_dir)


def main():
  args = cmd_args.parse_args()
  retrain_setup_logging(args)

  random.seed(args.retrain_rand_seed)
  np.random.seed(args.retrain_rand_seed)
  torch.manual_seed(args.retrain_rand_seed)
  torch.cuda.manual_seed_all(args.retrain_rand_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  train_loader, valid_loader, test_loader = get_data_loaders(args, shuffle_train=True)
  model = get_model(args)
  logging.info('Number of parameters: %d', sum([p.data.nelement() for p in model.parameters()]))

  #--------------------------------------------------------------------------
  # Retraining code
  #--------------------------------------------------------------------------
  exp_dir = os.path.join('runs', args.exp_name)
  checkpoint = torch.load(exp_dir + '/model_e' + args.load_epoch + '.t7')
  model.load_state_dict(checkpoint['state_dict'])

  direction = net_plotter_seed.create_random_direction(model, seed = args.retrain_rand_seed)
  weights = net_plotter_seed.get_weights(model)
  net_plotter_seed.set_weights(model, weights, directions = direction, step = args.step_size)
  retrain_model(args, model, train_loader, valid_loader, test_loader)



if __name__ == '__main__':
  main()
