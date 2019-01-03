import argparse
import datetime
import functools
import os
import random

import torch

import bf
from bf.builders import train_builder, data_builder
from bf.training import callbacks, helpers
from bf.utils.config_wrapper import ConfigWrapper
from detection.init import init as init_detection
from detection.metrics.mean_average_precision import mean_average_precision


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.py')
    parser.add_argument('--checkpoint', type=str, default='./experiments')
    parser.add_argument('--phases', nargs='+', default=['train', 'val'])
    args = parser.parse_args()

    cfg = helpers.load_config(args.config)
    cfg = ConfigWrapper(cfg)
    cfg.set_phases(args.phases)

    state = helpers.load_checkpoint(args.checkpoint)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    use_cuda = cfg.use_gpu and torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'

    dataloader, num_classes, dataset = data_builder.create_dataloaders(cfg.dataset,
                                                                       cfg.batch_size,
                                                                       cfg.input_size,
                                                                       cfg.augmentations,
                                                                       cfg.preprocessing,
                                                                       shuffle=cfg.shuffle,
                                                                       num_workers=cfg.num_workers,
                                                                       pin_memory=use_cuda)
    model, init_epoch_state_fn, step_fn = init_detection(device=device,
                                                         num_classes=num_classes,
                                                         model_params=cfg.model,
                                                         box_coder_params=cfg.box_coder,
                                                         postprocess_params=cfg.postprocess,
                                                         sampler_params=cfg.sampler,
                                                         loss_params=cfg.loss,
                                                         target_assigner_params=cfg.target_assigner,
                                                         state=state)
    print(model)

    if 'val' in args.phases:
        metrics = {'mAP': functools.partial(mean_average_precision,
                                            class_labels=dataset['val'].class_labels,
                                            iou_threshold=.5,
                                            voc=cfg.is_voc('val'))}
    else:
        metrics = {}

    if 'train' in args.phases:
        epochs = cfg.train['epochs']
        total_train_steps = len(dataloader['train']) // cfg.train['accumulation_steps']

        cfg.update(locals())

        optimizer = train_builder.create_optimizer(model, cfg.train['optimizer'], state=state)
        print(optimizer)

        if state:
            checkpoint_dir = args.checkpoint
        else:
            checkpoint_dir = os.path.join(args.checkpoint, f'{datetime.datetime.today():%F-%H%M%S}')

        trainer = bf.train.Trainer(epochs,
                                   args.phases,
                                   model,
                                   optimizer,
                                   init_epoch_state_fn=init_epoch_state_fn,
                                   step_fn=step_fn,
                                   accumulation_steps=cfg.train['accumulation_steps'],
                                   metrics=metrics,
                                   eval_every=cfg.train['eval_every'])

        callbacks.checkpoint(trainer, checkpoint_dir, config_path=args.config, save_every=cfg.train['eval_every'])
        callbacks.logger(trainer, checkpoint_dir)
        writer = callbacks.tensorboard(trainer, checkpoint_dir)

        if 'scheduler' in cfg.train:
            scheduler = train_builder.create_scheduler(cfg.train['scheduler'], optimizer, state=state)

            callbacks.scheduler(trainer, *scheduler)

            @trainer.on('scheduler_step')
            def log_lr(*args, **kwargs):
                for i, x in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'lr/Learning Rate {i}', x['lr'], trainer.global_step)

        if state:
            print(f'>> Resuming from: step: {state["global_step"]}, epoch: {state["epoch"]}')
            trainer.resume(initial_step=state['global_step'] + 1, initial_epoch=state['epoch'] + 1)

        trainer.run(dataloader)

    elif 'val' in args.phases:
        evaluator = bf.eval.Evaluator(model, init_epoch_state_fn, step_fn, metrics=metrics)
        evaluator.run(dataloader['val'])
