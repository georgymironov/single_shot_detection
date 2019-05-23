import functools

import bf
from bf.builders import train_builder, data_builder
from bf.training import callbacks, env, helpers
from bf.training.pruning.pruner import Pruner
from bf.utils.video_viewer import VideoViewer
from bf.utils import mo_exporter, onnx_exporter

from detection.init import init as init_detection
from detection.metrics.mean_average_precision import mean_average_precision
from detection.tools import mo_add_output


def main(args):
    env.init_logger(args)

    state, checkpoint_dir = helpers.init_checkpoint(args)
    env.init_file_logger(args, checkpoint_dir)
    cfg = helpers.load_config(args)

    env.init_random_state(args, cfg)
    device, use_cuda = env.set_device(args, cfg)

    augment, preprocess = data_builder.create_preprocessing(cfg.augmentations, cfg.preprocessing, cfg.input_size, 'box')

    if 'train' in args.phases or 'eval' in args.phases:
        datasets = data_builder.create_datasets(cfg.dataset, augment=augment, preprocess=preprocess)
        dataloaders = data_builder.create_dataloaders(datasets=datasets,
                                                      batch_size=cfg.batch_size,
                                                      shuffle=cfg.shuffle,
                                                      num_workers=cfg.num_workers,
                                                      pin_memory=use_cuda,
                                                      distributed=args.distributed)

    detector, init_epoch_state_fn, step_fn = init_detection(device=device,
                                                            model_params=cfg.model,
                                                            box_coder_params=cfg.box_coder,
                                                            postprocess_params=cfg.postprocess,
                                                            sampler_params=cfg.sampler,
                                                            loss_params=cfg.loss,
                                                            target_assigner_params=cfg.target_assigner,
                                                            state=state,
                                                            preprocess=preprocess,
                                                            distributed=args.distributed)

    if 'eval' in args.phases:
        metrics = {'mAP': functools.partial(mean_average_precision,
                                            class_labels=datasets['eval'].class_labels,
                                            iou_threshold=.5,
                                            voc=cfg.is_voc('val'))}
    else:
        metrics = {}

    if 'embed' in args.phases:
        import IPython
        IPython.embed()
        return

    if 'train' in args.phases:
        cfg.update({
            'epochs': cfg.train['epochs'],
            'total_train_steps': len(dataloaders['train']) // cfg.train.get('accumulation_steps', 1)
        })

        optimizer = train_builder.create_optimizer(detector.model, cfg.train['optimizer'], state=state)

        trainer = bf.train.Trainer(cfg.train['epochs'],
                                   args.phases,
                                   detector.model,
                                   init_epoch_state_fn=init_epoch_state_fn,
                                   step_fn=step_fn,
                                   accumulation_steps=cfg.train.get('accumulation_steps', 1),
                                   metrics=metrics,
                                   eval_every=cfg.train['eval_every'])

        event_emitter = trainer.event_emitter

        callbacks.optimizer(event_emitter, optimizer)
        callbacks.progress(event_emitter)
        callbacks.checkpoint(event_emitter, checkpoint_dir, save_every=cfg.train.get('eval_every', 1))
        callbacks.csv_logger(event_emitter, csv_log_path=helpers.get_csv_log_file(args, checkpoint_dir))

        if args.tensorboard:
            writer = callbacks.tensorboard(event_emitter, checkpoint_dir)
        else:
            writer = None

        if 'scheduler' in cfg.train:
            scheduler = train_builder.create_scheduler(cfg.train['scheduler'], optimizer, state=state)
            callbacks.scheduler(event_emitter, *scheduler, writer=writer)

        if 'pruner' in cfg.train:
            pruner = Pruner(detector.model, **cfg.train['pruner'])

            @event_emitter.on('epoch_start')
            def prune(*args, **kwargs):
                pruner.prune()

        if state:
            trainer.resume(initial_step=state['global_step'] + 1, initial_epoch=state['epoch'] + 1)

        trainer.run(dataloaders, num_batches_per_epoch=cfg.train.get('num_batches_per_epoch'))

    elif 'eval' in args.phases:
        evaluator = bf.eval.Evaluator(detector.model, init_epoch_state_fn, step_fn, metrics=metrics)
        evaluator.run(dataloaders['eval'])

    if 'test' in args.phases:
        viewer = VideoViewer(args.video, detector)
        viewer.run()

    if 'export' in args.phases:
        onnx_exporter.export(detector.model, cfg.input_size, 'model.onnx')

    if 'export-mo' in args.phases:
        mo_exporter.export(detector.model, cfg, 'model', folder='exported', postprocess=mo_add_output.add_output)


if __name__ == '__main__':
    parser = helpers.get_default_argparser()
    parser.add_argument('--phases', nargs='+', default=['train', 'eval'], choices=['train', 'eval', 'test', 'export', 'export-mo', 'embed'],
                        help='One or multiple runtime phases')
    args = parser.parse_args()

    helpers.launch(args, main)
