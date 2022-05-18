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
import datetime

from torchvision.datasets import CIFAR10
from cifar10_data import CIFAR10RandomLabels
from torch.utils.data.sampler import SubsetRandomSampler

import cmd_args
import model_mlp, model_wideresnet

import net_plotter_seed
import phate
from gtda.homology import VietorisRipsPersistence
from vietoris_rips_filtration import p_norm
import json
import igraph
from pyper.persistent_homology.graphs import extend_filtration_to_edges, calculate_persistence_diagrams

from phate import PHATE

def get_model(args):
  # create model
  random.seed(args.rand_seed)
  np.random.seed(args.rand_seed)
  torch.manual_seed(args.rand_seed)
  torch.cuda.manual_seed_all(args.rand_seed)
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

def get_model_params(model):
    for count, params in enumerate(model.parameters()):
        if count == 0:
            net_params = torch.flatten(params.data).cpu().numpy()
        else:
            net_params = np.append(net_params,torch.flatten(params.data).cpu().numpy())
    return net_params

def get_knn_connectivity_from_distance_matrix(distance_matrix, k):
    knn_list = []

    for line_idx, line in enumerate(distance_matrix):
        knn_indices = np.argsort(line)[-1:-(1+k):-1]
        knn_list.extend([(line_idx, idx) for idx in knn_indices])

    return knn_list

def main():
  args = cmd_args.parse_args()
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # Saving and Loading Directory
  save_dir = os.path.join('analysis_data', args.exp_name)
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

  # Load all parameter samples
  exp_dir = os.path.join('runs', args.exp_name)
  checkpoint = torch.load(exp_dir + '/model_e' + args.load_epoch + '.t7')
  model = get_model(args)
  model.load_state_dict(checkpoint['state_dict'])
  all_params_samples = get_model_params(model)

  retrain_vect_log_data = {'train_acc': [], 'train_loss': [], 'valid_acc': [], 'valid_loss': [], 'test_acc': [], 'test_loss': []}
  retrain_vect_log_data['train_acc'].append(checkpoint['train_accuracy'])
  retrain_vect_log_data['train_loss'].append(checkpoint['train_loss'])
  retrain_vect_log_data['valid_acc'].append(checkpoint['train_accuracy'])
  retrain_vect_log_data['valid_loss'].append(checkpoint['train_loss'])
  retrain_vect_log_data['test_acc'].append(checkpoint['test_accuracy'])
  retrain_vect_log_data['test_loss'].append(checkpoint['test_loss'])

  seeds = [0, 1, 2, 3]
  step_sizes = [0.25, 0.5, 0.75, 1.0]

  for seed in seeds:
      for step_size in step_sizes:
          retrain_name = cmd_args.format_retrain_name(args, seed=seed, step_size = step_size)
          retrain_dir = os.path.join('runs', args.exp_name, retrain_name)

          for retrain_epoch in range(40):
              if retrain_epoch == 0:
                  checkpoint = torch.load(retrain_dir + '/model_e0.t7')
                  model.load_state_dict(checkpoint['state_dict'])
                  all_params_samples = np.vstack((all_params_samples, get_model_params(model)))
              else:
                  checkpoint = torch.load(retrain_dir + '/model_e{0}_end.t7'.format(retrain_epoch))
                  model.load_state_dict(checkpoint['state_dict'])
                  all_params_samples = np.vstack((all_params_samples, get_model_params(model)))

              retrain_vect_log_data['train_acc'].append(checkpoint['train_accuracy'])
              retrain_vect_log_data['train_loss'].append(checkpoint['train_loss'])
              retrain_vect_log_data['valid_acc'].append(checkpoint['valid_accuracy'])
              retrain_vect_log_data['valid_loss'].append(checkpoint['valid_loss'])
              retrain_vect_log_data['test_acc'].append(checkpoint['test_accuracy'])
              retrain_vect_log_data['test_loss'].append(checkpoint['test_loss'])

  # Save the loss and accuracy data
  sampling_data = retrain_vect_log_data
  json.dump(sampling_data, open(os.path.join(save_dir,'vectorized_loss_samples.json'),'w'))

  # PHATE embeds and diffusion potentials
  phate_cos = PHATE(n_components = 2, knn_dist = 'cosine', mds_dist = 'cosine')
  np.save(os.path.join(save_dir,'phate_cos_2D_embeds.npy'), phate_cos.fit_transform(all_params_samples[1:]))
  phate_cos_diff_pot = phate_cos.diff_potential
  np.save(os.path.join(save_dir,'phate_cos_diff_pot.npy'), phate_cos_diff_pot)

  del all_params_samples

  # Compute graph and filtration
  knn = 20
  value = 'test_loss'
  order = 'sublevel'

  knn_list = get_knn_connectivity_from_distance_matrix(phate_cos_diff_pot, knn)
  graph = igraph.Graph(knn_list)
  graph.vs['f'] = sampling_data[value]

  graph_edges = extend_filtration_to_edges(graph, order = order)
  pers_diags = calculate_persistence_diagrams(graph_edges, order = order)
  dim0 = np.hstack([np.array(pers_diags[0]),np.zeros([len(np.array(pers_diags[0])),1])])
  dim1 = np.hstack([np.array(pers_diags[1]),np.ones([len(np.array(pers_diags[1])),1])])
  pers_diag = np.vstack([dim0,dim1])

  np.save(os.path.join(save_dir,'pers_diag_phate_cos_diff_pot_knn{}'.format(knn)+'_'+value+'_'+order+'.npy'), pers_diag)

if __name__ == '__main__':
  main()
