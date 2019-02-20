import functools
import random

import torch

import bf
from bf.builders import train_builder, data_builder
from bf.training import callbacks, helpers
from bf.training.pruner import Pruner
from bf.utils.video_viewer import VideoViewer
from bf.utils import mo_exporter, onnx_exporter

from detection.init import init as init_detection
from detection.metrics.mean_average_precision import mean_average_precision
from detection.tools import mo_add_output


if __name__ == '__main__':
    parser = helpers.get_default_argparser()
    args = parser.parse_args()

    helpers.init_logger(args)
    state, checkpoint_dir = helpers.init_checkpoint(args)
    cfg = helpers.load_config(args)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    use_cuda = cfg.use_gpu and torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'

    resize, augment, preprocess = data_builder.create_preprocessing(cfg.input_size, cfg.augmentations, cfg.preprocessing, 'box')

    if 'train' in args.phases or 'eval' in args.phases:
        dataloader, dataset = data_builder.create_dataloaders(cfg.dataset,
                                                              cfg.batch_size,
                                                              resize,
                                                              augment,
                                                              preprocess,
                                                              shuffle=cfg.shuffle,
                                                              num_workers=cfg.num_workers,
                                                              pin_memory=use_cuda)

    detector, init_epoch_state_fn, step_fn = init_detection(device=device,
                                                            model_params=cfg.model,
                                                            box_coder_params=cfg.box_coder,
                                                            postprocess_params=cfg.postprocess,
                                                            sampler_params=cfg.sampler,
                                                            loss_params=cfg.loss,
                                                            target_assigner_params=cfg.target_assigner,
                                                            state=state,
                                                            preprocess=preprocess,
                                                            resize=resize)

    if 'eval' in args.phases:
        metrics = {'mAP': functools.partial(mean_average_precision,
                                            class_labels=dataset['eval'].class_labels,
                                            iou_threshold=.5,
                                            voc=cfg.is_voc('val'))}
    else:
        metrics = {}

    if 'embed' in args.phases:
        import IPython
        IPython.embed()
    elif 'train' in args.phases:
        epochs = cfg.train['epochs']
        total_train_steps = len(dataloader['train']) // cfg.train.get('accumulation_steps', 1)

        cfg.update(locals())

        optimizer = train_builder.create_optimizer(detector.model, cfg.train['optimizer'], state=state)

        trainer = bf.train.Trainer(epochs,
                                   args.phases,
                                   detector.model,
                                   optimizer,
                                   init_epoch_state_fn=init_epoch_state_fn,
                                   step_fn=step_fn,
                                   accumulation_steps=cfg.train.get('accumulation_steps', 1),
                                   metrics=metrics,
                                   eval_every=cfg.train['eval_every'])

        if not args.debug:
            callbacks.checkpoint(trainer, checkpoint_dir, config_path=args.config, save_every=cfg.train['eval_every'])
            callbacks.logger(trainer, checkpoint_dir)

        writer = callbacks.tensorboard(trainer, checkpoint_dir) if args.tensorboard else None

        if 'scheduler' in cfg.train:
            scheduler = train_builder.create_scheduler(cfg.train['scheduler'], optimizer, state=state)

            callbacks.scheduler(trainer, *scheduler)

            if writer:
                @trainer.on('scheduler_step')
                def log_lr(*args, **kwargs):
                    for i, x in enumerate(optimizer.param_groups):
                        writer.add_scalar(f'lr/learning_rate_{i}', x['lr'], trainer.global_step)

        if 'pruner' in cfg.train:
            pruner = Pruner(detector.model, **cfg.train['pruner'])

            @trainer.on('epoch_start')
            def prune(*args, **kwargs):
                pruner.prune()

        if state:
            trainer.resume(initial_step=state['global_step'] + 1, initial_epoch=state['epoch'] + 1)

        trainer.run(dataloader, num_batches_per_epoch=cfg.train.get('num_batches_per_epoch'))

    elif 'eval' in args.phases:
        evaluator = bf.eval.Evaluator(detector.model, init_epoch_state_fn, step_fn, metrics=metrics)
        evaluator.run(dataloader['eval'])
    elif 'test' in args.phases:
        viewer = VideoViewer(args.video, detector)
        viewer.run()
    elif 'export' in args.phases:
        onnx_exporter.export(detector.model, cfg.input_size, 'model.onnx')
    elif 'export-mo' in args.phases:
        mo_exporter.export(detector.model, cfg, 'model', folder='exported', postprocess=mo_add_output.add_output)
