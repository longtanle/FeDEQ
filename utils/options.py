import os
import argparse

import pprint
import random
import sys
import time

def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer',
                        help='algorithm to run',
                        type=str,
                        choices=('fedavg', 'fedavg_resnet', 'fedavg_seq', 
                                 'fedper_resnet', 'fedper_seq',
                                 'fedrep_resnet', 'fedrep_seq',
                                 'knnper_resnet', 'knnper_seq',
                                 'fedeq_resnet', 'fedeq_seq',
                                 'local_resnet', 'local_seq', 
                                 'finetune_resnet', 'finetune_seq', 
                                 'ditto_resnet', 'ditto_seq'),
                        default='fedeq_vision')
    # Model                       
    parser.add_argument('-m', '--model',
                        help='type of model',
                        choices=('dnn_3l', 'dnn_5l', 'dnn_10l',                                
                                 'resnet34', 'resnet32', 'resnet14', 
                                 'transformer4', 'transformer8', 'transformer12',
                                 'deq_mlp', 'deq_mlp_v2', 
                                 'deq_resnet_softplus', 'deq_resnet_relu', 
                                 'deq_resnet_s', 'deq_resnet_m',
                                 'deq_transformer'),
                        type=str,
                        default='deq_resnet_s')
    # Client                       
    parser.add_argument('-c', '--num_clients',
                        help='type of model',
                        choices=(1, 10, 20, 50, 
                                 100, 150, 
                                 200, 300, 
                                 500, 715,
                                 1000, 2000, 3000),
                        type=int,
                        default=100)
    # Datasets
    parser.add_argument('-d', '--dataset',
                        help='name of dataset',
                        choices=('femnist', 
                                 'cifar10', 
                                 'cifar100',
                                 'shakespeare'),
                        type=str,
                        required=True)
    parser.add_argument('-lpc', '--labels_per_client',
                        help='the number of labels per client (for Cifar dataset)',
                        type=int,
                        default=5)    
    parser.add_argument('--linf_proj',
                        help='using l-infinity projection',
                        type=int,
                        default=0)    
    parser.add_argument('--density',
                        type=float,
                        help='Fraction of the local training data to use (for each silo)',
                        default=1.0)
    parser.add_argument('--no_std',
                        help='Disable dataset standardization (vehicle and gleam only)',
                        action='store_true')
    # Learning
    parser.add_argument('-t', '--num_rounds',
                        help='number of communication rounds',
                        type=int,
                        default=400)
    parser.add_argument('--seed',
                        help='root seed for randomness',
                        type=int,
                        default=0)
    parser.add_argument('-lr', '--learning_rate',
                        help='client learning rate for local training',
                        type=float,
                        default=0.01)
    parser.add_argument('--lrs',
                        help='sweep client learning rate',
                        nargs='+',
                        type=float)
    parser.add_argument('--lambda',
                        help='parameter for personalization',
                        type=float,
                        default=0.3)
    parser.add_argument('--lambdas',
                        help='sweep lambda values',
                        nargs='+',
                        type=float)
    parser.add_argument('--l2_reg',
                        help='L2 regularization',
                        type=float,
                        default=0.0)
    parser.add_argument('--lam_svm',
                        help='regularization parameter for linear SVM',
                        type=float,
                        default=0.0)  # this param is kept the same for all methods and for all runs
    parser.add_argument('-ee', '--eval_every',
                        help='evaluate every `eval_every` rounds;',
                        type=int,
                        default=1)
    parser.add_argument('-cr', '--clients_per_round',
                        help='number of clients trained per round; -1 means use all clients',
                        type=float,
                        default=0.1)
    parser.add_argument('-b', '--batch_size',
                        help='batch size for client optimization',
                        type=int,
                        default=10)
    parser.add_argument('-pb', '--per_batch_size',
                        help='personalization batch size for client optimization',
                        type=int,
                        default=-1)                    
    parser.add_argument('-im', '--inner_mode',
                        help='How to run inner loop (fixed no. of batches or epochs)',
                        type=str,
                        choices=('iter', 'epoch'),
                        default='epoch')
    parser.add_argument('-le', '--inner_epochs',
                        help='number of epochs per communication round',
                        type=int,
                        default=5)
    parser.add_argument('-pe', '--personalized_epochs',
                        help='number of personalization epochs per communication round',
                        type=int,
                        default=3)                    
    parser.add_argument('-fu', '--frac_unseen',
                        help='Fraction of unseen clients',
                        type=float,
                        default=0.0)    
    parser.add_argument('-ii', '--inner_iters',
                        help='number of inner iterations per communication round',
                        type=int,
                        default=1)
    parser.add_argument('--unweighted_updates',
                        help='Disable weighing client model updates by their example counts',
                        action='store_true')
    # Server optimizers
    parser.add_argument('--server_lr',
                        help='learning rate for server opt',
                        type=float,
                        default=1)    
    parser.add_argument('--server_opt',
                        help='Server optimizer',
                        choices=('fedavg', 'fedavgm', 'fedadam'),
                        type=str,
                        default='fedavg')
    parser.add_argument('--fedavg_momentum',
                        help='momentum for FedAvgM',
                        type=float,
                        default=0.9)
    parser.add_argument('--fedadam_beta1',
                        help='Beta_1 for FedAdam',
                        type=float,
                        default=0.9)
    parser.add_argument('--fedadam_beta2',
                        help='Beta_2 for FedAdam',
                        type=float,
                        default=0.99)
    parser.add_argument('--fedadam_tau',
                        help='Tau for FedAdam (term in denominator)',
                        type=float,
                        default=1e-3)    
    # DEQ                       
    parser.add_argument('-fs', '--fwd_solver',
                        help='DEQ forward solver',
                        choices=('anderson', 'fpi', 'broyden', 'spsa', 'ra'),
                        type=str,
                        default='anderson')

    parser.add_argument('-bs', '--bwd_solver',
                        help='DEQ backward solver',
                        choices=('normal_cg', 'gmres'),
                        type=str,
                        default='normal_cg')
    # ADMM
    parser.add_argument('--rho',
                        help='Rho value for ADMM quadratic term',
                        type=float,
                        default=0.01)

    parser.add_argument('--lam_admm',
                        help='lamda value for updating ADMM lambda',
                        type=float,
                        default=0.01)                        

    # Finetuning
    parser.add_argument('--finetune_frac',
                        type=float,
                        help='Fraction of rounds for fedavg training',
                        default=0.3)
    
    # Knn-Per
    parser.add_argument('--knn_neighbors',
                        help='Number of neighbors for kNN-Per',
                        type=int,
                        default=10)
    parser.add_argument('--knn_weights',
                        help='Weights for the neighbors when making predictions for kNN',
                        type=str,
                        choices=('uniform', 'distance'),
                        default='distance')
    parser.add_argument('-kl','--knn_lam',
                        help='Personalization coefficient for kNN-Per',
                        type=float,
                        default=0.5)
    # Misc args
    parser.add_argument('-o', '--outdir',
                        help=('Directory to store artifacts, under `logs/`.'),
                        type=str)
    parser.add_argument('-r', '--repeat',
                        help=('Number of times to repeat the experiment'),
                        type=int,
                        default=1)
    parser.add_argument('-q', '--quiet',
                        help='Try not to print things',
                        action='store_true')
    parser.add_argument('--no_per_round_log',
                        help='Disable storing eval metrics',
                        action='store_true')
    parser.add_argument('--num_procs',
                        help='number of parallel processes for mp.Pool()',
                        type=int)
    parser.add_argument('--downsize_pool',
                        help='Downsize the multiprocessing pool',
                        action='store_true')
    parser.add_argument('-g', '--gpu',
                        help='GPU ID',
                        type=str,
                        default='0')
    parser.add_argument('-wa', '--wandb',
                        help='Wandb Logging',
                        type=int,
                        default=0)

    args = parser.parse_args()
    print(f'Command executed: python3 {" ".join(sys.argv)}')

    if args.outdir is None:
      print(f'Outdir not provided.', end=' ')
      args.outdir = f'logs/{args.trainer}--{time.strftime("%Y-%m-%d--%H-%M-%S")}--{args.dataset}-{args.num_clients}-{args.model}'
    os.makedirs(args.outdir, exist_ok=True)
    print(f'Storing outputs to {args.outdir}')

    if args.seed is None or args.seed < 0:
      print(f'Random seed not provided.', end=' ')
      args.seed = random.randint(0, 2**32 - 1)
    print(f'Using {args.seed} as global seed.')

    # Problem types
    args.is_regression = (args.dataset in ('school', 'adni'))
    args.is_linear = (args.dataset in ('mnist', 'femnist', 'cifar', 'tiny_imgnet'))

    # Record flags and input command.
    # NOTE: the `args.txt` would NOT include the parallel sweep hparams (e.g. lambdas).
    parsed = vars(args)
    with open(os.path.join(args.outdir, 'args.txt'), 'w') as f:
      pprint.pprint(parsed, stream=f)
    with open(os.path.join(args.outdir, 'command.txt'), 'w') as f:
      print(' '.join(sys.argv), file=f)

    print(parsed)
    return parsed