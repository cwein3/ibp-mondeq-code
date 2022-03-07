import os
import argparse as ap 

from data import mnist_loaders, cifar_loaders, mnist_transform, cifar_transform
from train import get_model, train, resume_killed
from model.imp_models import MON_NAMES
from model.ff_models import FF_NAMES
import logger


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--dataset', 
        choices=['mnist', 'cifar'], 
        help='Dataset choices.') 
    parser.add_argument(
        '--save_dir',
        type=str,
        help='Where to save the runs.')
    parser.add_argument(
        '--data_cache_dir',
        type=str,
        help='Where to download and cache MNIST, SVHN, CIFAR datasets.')

    parser.add_argument(
        '--arch',
        choices=MON_NAMES + FF_NAMES,
        help='Model architecture.')
    parser.add_argument(
        '--explicit_7_additional',
        type=int, 
        default=0,
        help='Number of additional layers for 7 layer explicit architecture.')
    parser.add_argument(
        '--out_channels',
        type=int,
        default=128,
        help='Number of channels to have for 3 layer architectures.'
        )
    parser.add_argument(
        '--splitting',
        choices=['fb', 'fb_anderson'],
        default='fb_anderson',
        help='Which method to use to solve for equilibrium.')
    parser.add_argument(
        '--lben',
        action='store_true',
        help='Whether to have LBEN parameterization.')
    parser.add_argument(
        '--lben_cond',
        type=float,
        default=3,
        help='Condition number of diagonal matrix in LBEN parameterization.')
    
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=10, 
        help='Clip the gradient norms to this threshold.')
    parser.add_argument(
        '--lr_mode',
        choices=['step', 'constant'],
        default='step',
        help='Choice of learning rate schedule.')
    parser.add_argument(
        '--step',
        type=int,
        nargs='+',
        default=[120, 280],
        help='Epochs during which to anneal learning rate.')
    parser.add_argument(
        '--anneal_factor',
        type=float,
        default=0.2,
        help='Factor by which to anneal the learning rate.')
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-4,
        help='Max LR to use.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size.')
    parser.add_argument(
        '--crop_pad',
        type=int,
        default=2,
        help='How much to crop for CIFAR data augmentation.')

    parser.add_argument(
        '--tune_alpha',
        action='store_true',
        help='Whether to tune the alpha parameter in MonDEQ solver.')
    parser.add_argument(
        '--max_alpha',
        type=float,
        default=1.0,
        help='Maximum value of alpha parameter in MonDEQ solver.')
    parser.add_argument(
        '--m',
        type=float,
        default=0.5,
        help='Monotonicity parameter.')
    parser.add_argument(
        '--m_init',
        type=float,
        default=0.99,
        help='Initial value of m.')
    parser.add_argument(
        '--anneal_m',
        action='store_true',
        help='Whether to anneal m over the course of training.')

    parser.add_argument(
        '--no_ibp',
        action='store_true',
        help='If true, do not use IBP.')
    parser.add_argument(
        '--ibp_init',
        action='store_true',
        help='Whether to initialize weights according to IBP init scheme.')

    parser.add_argument(
        '--eps',
        type=float,
        default=0.00785,
        help='Epsilon for l_infinity perturbation during training.')
    parser.add_argument(
        '--eps_warmup',
        type=int,
        default=1,
        help='Number of epochs to have epsilon = 0 as warmup.')
    parser.add_argument(
        '--eps_ramp',
        type=int,
        default=79,
        help='Number of epochs with which to ramp up the value of epsilon.')
    parser.add_argument(
        '--test_eps',
        type=float,
        default=0.00785,
        help='Epsilon used during testing.')

    parser.add_argument(
        '--log_every',
        type=int,
        default=50,
        help='How often to log.')
    parser.add_argument(
        '--ckpt_every',
        type=int,
        default=20,
        help='How often to store checkpoints.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
        help='How many epochs to train for.')

    parser.add_argument(
        '--slurm_dir',
        type=str,
        help='Slurm dir for run.')

    parser.add_argument(
        '--resume_killed',
        action='store_true',
        help='Whether to resume from dead job.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    args.ibp = not args.no_ibp

    if args.resume_killed:
        args.resume = resume_killed(os.path.join(args.save_dir, args.slurm_dir))
    else:
        args.resume = None

    if args.dataset == 'mnist':
        train_loader, test_loader = mnist_loaders(args.batch_size, args, test_batch_size=args.batch_size)
        normalize = mnist_transform()
    elif args.dataset == 'cifar':
        train_loader, test_loader = cifar_loaders(args.batch_size, args, test_batch_size=args.batch_size, crop_pad=args.crop_pad)
        normalize = cifar_transform()

    model = get_model(normalize, args)
    print(model)

    log_writer = logger.ExperimentLogWriter(
        args.save_dir,
        args.slurm_dir)

    print('Args:', args)
    log_writer.save_args(args)

    train(train_loader, test_loader, model, log_writer, args)

if __name__ == '__main__':
    main()
