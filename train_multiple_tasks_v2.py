#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import numpy as np
import os
import sys
import random

import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def cross_validate(stats, no_training):
    best_train_accuracies = []
    val_accuracies = []
    task_val_accuracies_all = []

    num_tasks = len(stats)

    for i in range(num_tasks):
        train_accs, val_accs = stats[i]
        task_train_accuracies = [float(train_accs[i]['ppl']) for i in range(len(train_accs))]
        task_val_accuracies = [float(val_accs[i]['ppl']) for i in range(len(val_accs))]
        task_val_accuracies_all.append(task_val_accuracies)

        best_train_accuracies.append(min(task_train_accuracies))
        val_accuracies.append(min(task_val_accuracies))

    task_val_accuracies_mean = np.mean(task_val_accuracies_all, axis=0)
    best_num_iters = np.argmin(task_val_accuracies_mean)

    val_accuracy = task_val_accuracies_mean[best_num_iters] 
    best_train_accuracy = sum(best_train_accuracies) / len(best_train_accuracies)

    return val_accuracy, best_train_accuracy, best_num_iters


def main(args, examples, state=None, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    # Print args
    print(args)
    
    task = tasks.setup_task(args, load_data=False)
    task.examples = examples

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=True, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    print('Mode', model.training_mode)

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    if state is not None:
        extra_state, epoch_itr = checkpoint_utils.load_checkpoint_from_state(args, trainer, state)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')

    train_stats_all = []
    valid_stats_all = []

    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train_stats = train(args, trainer, task, epoch_itr)
        train_stats_all.append(train_stats)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses, valid_stats = validate(args, trainer, task, epoch_itr, valid_subsets)
            valid_stats_all.append(valid_stats)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if ':' in getattr(args, 'data', ''):
            # sharded data: get train iterator for next epoch
            epoch_itr = trainer.get_train_iterator(epoch_itr.epoch)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))

    return train_stats_all, valid_stats_all


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses, valid_stats = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    return stats


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats['loss'].avg)
    return valid_losses, stats


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        stats['best_loss'] = min(
            checkpoint_utils.save_checkpoint.best, stats['loss'].avg)
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main(args, examples, state):
    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        train_stats, valid_stats = main(args, examples, state=state)

    return train_stats, valid_stats


def master_main():

    stdout = sys.stdout
    f = open(os.devnull, 'w')
    sys.stdout = f

    parser = options.get_training_parser()
    parser.add_argument("--fast-eval", action="store_true", help="Fast eval mode.")
    parser.add_argument("--eval-num-iter", default=10, type=int, help="Number of eval training iterations.")
    args = options.parse_args_and_arch(parser)

    # Setup task, e.g., translation, language modeling, etc.
    task_gen = tasks.setup_task(args)
    examples = task_gen.examples

    no_training = args.no_training
    if no_training:
        args.max_epoch = 1

    if args.fast_eval:
        best_num_iter = args.eval_num_iter - 1
    else:
        restore_path = '/'.join(args.restore_file.split('/')[:-1])

        best_val = float('inf')

        for ckpt in range(10, 101, 10):
            args.restore_file = '%s/checkpoint%d.pt' % (restore_path, ckpt)
            print(args.restore_file)
            print(os.path.exists(args.restore_file))
            assert os.path.exists(args.restore_file)
            state = checkpoint_utils.load_checkpoint_to_cpu(args.restore_file)

            all_stats = []
            for task in range(16):
                args.eval_task_id = task

                train_stats, valid_stats = cli_main(args, examples, state)
                all_stats.append((train_stats, valid_stats))

            val, best_train, num_iter = cross_validate(all_stats, no_training)

            if val < best_val:
                best_val = val
                best_mdl = (val, best_train, num_iter, ckpt)

        val, best_train, best_num_iter, best_ckpt = best_mdl
        args.restore_file = '%s/checkpoint%d.pt' % (restore_path, best_ckpt)

    assert os.path.exists(args.restore_file)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.restore_file)

    train_accs, test_accs = [], []
    for task in range(16, 64):
        args.eval_task_id = task

        train_stats, valid_stats = cli_main(args, examples, state)
        train_accs.append(float(train_stats[best_num_iter]['ppl']))
        test_accs.append(float(valid_stats[best_num_iter]['ppl']))

    test = np.mean(test_accs)
    train = np.mean(train_accs)

    sys.stdout = stdout

    if args.fast_eval:
        print('test, train, best_num_iter')
        print('%.2f %.2f %d' % (test, train, best_num_iter))
    else:
        print('test, train, best_val, best_train, best_num_iter, best_ckpt')
        print('%s %d %.2f %.2f %.2f %.2f %d %d' % (restore_path, args.num_train, test, train, val, best_train, best_num_iter, best_ckpt))


if __name__ == '__main__':
    master_main()
