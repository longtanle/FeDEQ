#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import collections
import gc
import multiprocessing as mp
import os

import pprint
import random
import sys
import time

import numpy as np

from utils.options import read_options
from utils import data_utils

import jax
from jax.lib import xla_bridge
#from jax_smi import initialise_tracking
#initialise_tracking()

from trainers.fedavg import FedAvg
from trainers.vision.fedavg_resnet import FedAvg_Resnet
from trainers.sequence.fedavg_seq import FedAvg_SEQ
from trainers.vision.finetune_resnet import FinetuneResnet
from trainers.sequence.finetune_seq import FinetuneSEQ
from trainers.vision.fedrep_resnet import FedRep
from trainers.sequence.fedrep_seq import FedRep_SEQ
from trainers.vision.fedper_resnet import FedPer
from trainers.sequence.fedper_seq import FedPer_SEQ
from trainers.vision.fedeq_resnet import FeDEQ_ResNet
from trainers.sequence.fedeq_seq import FeDEQ_SEQ
from trainers.vision.knnper_resnet import kNNPer
from trainers.sequence.knnper_seq import kNNPer_SEQ
from trainers.vision.ditto_resnet import Ditto_ResNet
from trainers.sequence.ditto_seq import DittoSEQ
from trainers.vision.local_resnet import LocalResnet
from trainers.sequence.local_seq import LocalSEQ



def main(options, run_idx=None):

    options['run_idx'] = run_idx
    # set worker specific config.
    if run_idx is not None:
      options['seed'] += 1000 * run_idx
      options['outdir'] = os.path.join(options['outdir'], f'run{run_idx}')
      os.makedirs(options['outdir'], exist_ok=True)
      print(f'Run {run_idx} uses master seed {options["seed"]}')


    ###########################
    ##### Create Datasets #####
    ###########################

    seed = options['seed']
    random.seed(1 + seed)
    np.random.seed(12 + seed)
    dataset_args = dict(seed=seed, bias=False, density=options['density'],
                        standardize=(not options['no_std']))

    # Read data as ragged arrays with (K, n_i, ...).
    # Image datasets do not take dataset seed; randomness is for params/SGD.
    # The seed for datasets are fixed at data generation time.
    if options['dataset'] == 'femnist':
      dataset = data_utils.read_femnist_dataset(dataset=options['dataset'],
                                                  num_clients = options['num_clients'],
                                                  trainer = options['trainer'],
                                                  data_dir='data/femnist',
                                                  **dataset_args) 
    elif options['dataset'] == 'cifar10':
      dataset = data_utils.read_cifar10_data(num_clients = options['num_clients'],
                                              trainer = options['trainer'],
                                              num_labels = options['labels_per_client'],
                                              **dataset_args)
    elif options['dataset'] == 'cifar100' or options['dataset'] == 'cifar':
      dataset = data_utils.read_cifar100_data(num_clients = options['num_clients'],
                                              trainer = options['trainer'],
                                              num_labels = options['labels_per_client'],
                                              **dataset_args)
    elif options['dataset'] == 'shakespeare':
          dataset = data_utils.read_shakespeare_dataset(dataset=options['dataset'],
                                                      num_clients = options['num_clients'],
                                                      trainer = options['trainer'],
                                                      num_labels = options['labels_per_client'],
                                                      data_dir='data/shakespeare',
                                                      **dataset_args) 
    else:
        raise ValueError(f'Unknown dataset `{options["dataset"]}`')

    ###########################
    ##### Create Trainers #####
    ###########################

    if options['trainer'] == 'fedavg':
      t = FedAvg(options, dataset)
      result = t.train()

    elif options['trainer'] == 'fedavg_resnet':
      t = FedAvg_Resnet(options, dataset)
      result = t.train()

    elif options['trainer'] == 'fedavg_seq':
      t = FedAvg_SEQ(options, dataset)
      result = t.train()

    elif options['trainer'] == 'finetune_resnet':
      t = FinetuneResnet(options, dataset)
      result = t.train()  

    elif options['trainer'] == 'finetune_seq':
      t = FinetuneSEQ(options, dataset)
      result = t.train()

    elif options['trainer'] == 'local_resnet':
      t = LocalResnet(options, dataset)
      result = t.train()  

    elif options['trainer'] == 'local_seq':
      t = LocalSEQ(options, dataset)
      result = t.train()

    elif options['trainer'] == 'ditto_resnet':
      t = Ditto_ResNet(options, dataset)
      result = t.train()  

    elif options['trainer'] == 'ditto_seq':
      t = DittoSEQ(options, dataset)
      result = t.train()

    elif options['trainer'] == 'fedrep_resnet':
      t = FedRep(options, dataset)
      result = t.train()

    elif options['trainer'] == 'fedrep_seq':
      t = FedRep_SEQ(options, dataset)
      result = t.train()

    elif options['trainer'] == 'fedper_resnet':
      t = FedPer(options, dataset)
      result = t.train()

    elif options['trainer'] == 'fedper_seq':
      t = FedPer_SEQ(options, dataset)
      result = t.train()

    elif options['trainer'] == 'knnper_resnet':
      t = kNNPer(options, dataset)
      result = t.train()

    elif options['trainer'] == 'knnper_seq':
      t = kNNPer_SEQ(options, dataset)
      result = t.train()

    elif options['trainer'] == 'fedeq_resnet':
      t = FeDEQ_ResNet(options, dataset)
      result = t.train()

    elif options['trainer'] == 'fedeq_seq':
      t = FeDEQ_SEQ(options, dataset)
      result = t.train()

    else:
      raise ValueError(f'Unknown trainer `{options["trainer"]}`')

    # Run garbage collection to ensure finished runs don't keep unnecessary memory
    gc.collect()
    print(f'Outputs stored at {options["outdir"]}')
    return result


def repeat_main(options):
  num_repeats = options['repeat']
  with mp.Pool(num_repeats + 1) as pool:
    results = [pool.apply_async(main, (options, run_idx))
               for run_idx in range(num_repeats)]
    results = [r.get() for r in results]
    return results  # (num_repeats,)


def sweep_main(options):
    """Handles repeats, LR sweeps, and lambda sweeps."""
    options['no_per_round_log'] = True  # Disable per-round log since file size is too large.
    num_repeats = options['repeat']
    print(f'Sweeping over lams={options["lambdas"]}, lr={options["lrs"]}, repeat={num_repeats}')
    results = collections.defaultdict(list)

    def runner(lr, lam):
        cur_dir = f'{options["outdir"]}/lam{lam}_lr{lr}'
        cur_options = {**options, 'lambda': lam, 'learning_rate': lr, 'outdir': cur_dir}
        return [pool.apply_async(main, (cur_options, run_idx))
                for run_idx in range(num_repeats)]

    if options['downsize_pool']:
        print('Note: downsizing the multiprocessing pool along the lambda axis.')
        for lam in options['lambdas']:
            with mp.Pool(options['num_procs']) as pool:
                for lr in options['lrs']:
                    results[lr, lam] = runner(lr, lam)
                for lr in options['lrs']:
                    results[lr, lam] = [r.get() for r in results[lr, lam]]
    else:
        with mp.Pool(options['num_procs']) as pool:
            for lam in options['lambdas']:
                for lr in options['lrs']:
                    results[lr, lam] = runner(lr, lam)
            for lam in options['lambdas']:
                for lr in options['lrs']:
                    results[lr, lam] = [r.get() for r in results[lr, lam]]

    print(f'Sweep outputs stored at {options["outdir"]}')
    return results  # ((lrs, lams, repeats) of (train, test))

if __name__ == '__main__':
    options = read_options()
    print(f'outdir: {options["outdir"]}')

    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu'] 
    device = jax.devices()
    #print(device)
    #print("Running Device:", xla_bridge.get_backend().platform)
    print(f"[INFO] Using GPU {options['gpu']}")

    # Handle sweeping separately
    if options['lambdas'] is not None or options['lrs'] is not None:
        # Populate a sweep list if doesn't exist
        options['lrs'] = options['lrs'] or [options['learning_rate']]
        options['lambdas'] = options['lambdas'] or [options['lambda']]
        # Perform sweep and take stats over repertition
        out = sweep_main(options)
        for (lr, lam), repeat_vals in out.items():
            # Axis=0 to ensure taking stats for train/test separately
            out[lr, lam] = [np.mean(repeat_vals, axis=0), np.std(repeat_vals, axis=0)]

        # Rank best results differently for regression
        rank_fn = min if options['is_regression'] else max
        # Stats over lambda sweep
        lr_out, lam_out = {}, {}
        for lr in options['lrs']:   # output result for each LR.
            res = [out[lr, lam] for lam in options['lambdas']]
            lr_out[lr] = rank_fn(res, key=lambda x: x[0][1])   # Best run by the mean of test runs.
        for lam in options['lambdas']:   # output result for each lam.
            res = [out[lr, lam] for lr in options['lrs']]
            lam_out[lam] = rank_fn(res, key=lambda x: x[0][1])
        # Stats over all sweep; best run by the mean of test runs.
        best_hparams, best_run = rank_fn(dict(out).items(), key=lambda x: x[1][0][1])
        assert (np.array(best_run) ==
                np.array(rank_fn(lam_out.values(), key=lambda x: x[0][1]))).all()
        # Save results
        with open(os.path.join(options['outdir'], 'full_result.txt'), 'w') as f:
            pprint.pprint(dict(out), stream=f)
        with open(os.path.join(options['outdir'], 'best_result.txt'), 'w') as f:
            pprint.pprint({best_hparams: best_run}, stream=f)
        with open(os.path.join(options['outdir'], 'lr_sweep_lam_result.txt'), 'w') as f:
            pprint.pprint(lr_out, stream=f)
        with open(os.path.join(options['outdir'], 'lam_sweep_lr_result.txt'), 'w') as f:
            pprint.pprint(lam_out, stream=f)

    # No sweeping
    else:
        if options['repeat'] == 1:
            out = main(options)
        else:
            out = repeat_main(options)

        out = np.atleast_2d(out)
        stats = [np.mean(out, axis=0), np.std(out, axis=0)]
        print(f'final output:\n{pprint.pformat(out)}')
        print(f'mean, std:\n{stats}')
        if options['is_regression']:
            print(f'final test metric: {stats[0][1]:.5f} ± {stats[1][1]:.5f}')
        else:
            print(f'final test metric: {stats[0][1] * 100:.3f} ± {stats[1][1] * 100:.3f}')

        with open(os.path.join(options['outdir'], 'final_result.txt'), 'w') as f:
            pprint.pprint(out, stream=f)
            print(stats, file=f)

    print(f'Final outputs stored at {options["outdir"]}')
