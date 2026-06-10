import logging
import os
import sys
import numpy 
import random

from src.trainer import which
name, mem = '', 0
if which('nvidia-smi') is not None:
    _min_mem = 8000
    _deviceid = 0
    try:
        _line = os.popen('"nvidia-smi" --query-gpu=gpu_name,memory.total --format=csv,nounits,noheader').read().split('\n')[_deviceid]
        name, mem = _line.split(',')
        mem = int(mem)
        if mem < _min_mem:
            print(f'Less GPU memory ({mem} MB) than requested ({_min_mem} MB). Will try to continue.')
    except Exception:
        pass

logger = logging.getLogger('')

import torch
from torch.utils.data import DataLoader

from src.models import PELICANNano
from src.models import tests
from src.trainer import Trainer
from src.trainer import init_argparse, init_file_paths, init_logger, init_cuda, logging_printout, fix_args
from src.trainer import init_optimizer, init_scheduler
from src.models.metrics_classifier import metrics, minibatch_metrics, minibatch_metrics_string
from src.layers.quant import QuantConfig

from src.dataloaders import initialize_datasets, collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000, sci_mode=False)


def main():

    # Initialize arguments -- Just
    args = init_argparse()

    # Initialize file paths
    args = init_file_paths(args)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    # Initialize logger
    init_logger(args)

    if which('nvidia-smi') is not None:
        logger.info(f'Using {name} with {mem} MB of GPU memory')

    # Write input paramaters and paths to log
    logging_printout(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initialize dataloder
    if args.fix_data:
        torch.manual_seed(165937750084982)
    args, datasets = initialize_datasets(args, args.datadir, num_pts=None)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    # Construct PyTorch dataloaders from datasets
    collate = lambda data: collate_fn(data, scale=args.scale, nobj=args.nobj, add_beams=args.add_beams, beam_mass=args.beam_mass)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     worker_init_fn=seed_worker,
                                     collate_fn=collate)
                   for split, dataset in datasets.items()}

    # Build quantization config (disabled by default)
    quant_config = QuantConfig(
        enabled=args.quant,
        weight_bit_width=args.weight_bit_width,
        act_bit_width=args.act_bit_width,
        input_bit_width=args.input_bit_width,
        weight_per_channel=args.weight_per_channel,
        po2_scales=args.po2_scales,
        allow_alpha_scaling=args.allow_alpha_scaling,
    )

    # Initialize model
    model = PELICANNano(args.n_hidden,
                        activate_agg=args.activate_agg, activate_lin=args.activate_lin,
                        activation=args.activation, add_beams=args.add_beams, config=args.config, config_out=args.config_out, average_nobj=args.nobj_avg,
                        factorize=args.factorize, masked=args.masked,
                        activate_agg_out=args.activate_agg_out, activate_lin_out=args.activate_lin_out,
                        scale=args.scale, dropout=args.dropout, drop_rate=args.drop_rate, drop_rate_out=args.drop_rate_out, batchnorm=args.batchnorm,
                        quant_config=quant_config,
                        device=device, dtype=dtype)
    
    model.to(device)

    if args.parallel:
        model = torch.nn.DataParallel(model)

    # Initialize the scheduler and optimizer
    if args.task.startswith('eval'):
        optimizer = scheduler = None
        restart_epochs = []
        summarize = False
    else:
        optimizer = init_optimizer(args, model, len(dataloaders['train']))
        scheduler, restart_epochs = init_scheduler(args, optimizer)

    # Define a loss function.
    # loss_fn = torch.nn.functional.cross_entropy
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Apply the covariance and permutation invariance tests.
    if args.test:
        tests(model, dataloaders['train'], args, tests=['gpu','irc', 'permutation'])

    # Instantiate the training class
    trainer = Trainer(args, dataloaders, model, loss_fn, metrics, minibatch_metrics, minibatch_metrics_string, optimizer, scheduler, restart_epochs, args.summarize_csv, args.summarize, device, dtype)
    
    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Set a CUDA variale that makes the results exactly reproducible on a GPU (on CPU they're reproducible regardless)
    if args.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # Train model.
    if not args.task.startswith('eval'):
        trainer.train()

    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate(splits=['test'])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    main()
