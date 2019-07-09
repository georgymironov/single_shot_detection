import functools

import bf
from bf.builders import train_builder, data_builder
from bf.training import callbacks, env, helpers
from bf.training.pruning.pruner import Pruner
from bf.utils.video_viewer import VideoViewer
from bf.utils import mo_exporter, onnx_exporter, jit_exporter

from detection.init import init as init_detection
from detection.metrics.mean_average_precision import mean_average_precision
from detection.tools import mo_add_output


def main(args):
    env.init_logger(args)

    state, checkpoint_dir = helpers.init_checkpoint(args)
    env.init_file_logger(args, checkpoint_dir)
    cfg = helpers.load_config(args)

    env.set_random_state(args, cfg)
    device, use_cuda = env.set_device(args, cfg)

    augment, preprocess = data_builder.create_preprocessing(cfg.augmentations, cfg.preprocessing, cfg.input_size, 'box')

    if 'train' in args.phases or 'eval' in args.phases:
        datasets = data_builder.create_datasets(cfg.dataset, augment=augment, preprocess=preprocess)

        samplers = data_builder.create_samples(datasets=datasets,
                                               shuffle=cfg.shuffle,
                                               distributed=args.distributed)

        dataloaders = data_builder.create_dataloaders(datasets=datasets,
                                                      samplers=samplers,
                                                      batch_size=cfg.batch_size,
                                                      num_workers=cfg.num_workers,
                                                      pin_memory=use_cuda)

        if 'num_classes' not in cfg.model['detector']:
            cfg.model['detector']['num_classes'] = datasets['train'].num_classes if 'train' in args.phases else datasets['eval'].num_classes

    detector, init_epoch_state_fn, step_fn = init_detection(device=device,
                                                            model_args=cfg.model,
                                                            box_coder_args=cfg.box_coder,
                                                            postprocess_args=cfg.postprocess,
                                                            sampler_args=cfg.sampler,
                                                            loss_args=cfg.loss,
                                                            target_assigner_args=cfg.target_assigner,
                                                            state=state,
                                                            preprocess=preprocess,
                                                            parallel=args.parallel,
                                                            distributed=args.distributed,
                                                            mixup_args=cfg.mixup)

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

        if args.amp:
            helpers.init_amp(args, detector.model, optimizer)

        trainer = bf.train.Trainer(cfg.train['epochs'],
                                   args.phases,
                                   detector.model,
                                   init_epoch_state_fn=init_epoch_state_fn,
                                   step_fn=step_fn,
                                   accumulation_steps=cfg.train.get('accumulation_steps', 1),
                                   metrics=metrics,
                                   eval_every=cfg.train['eval_every'])

        event_emitter = trainer.event_emitter

        callbacks.loss(event_emitter, amp=args.amp)
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
        callbacks.progress(evaluator.event_emitter)
        evaluator.run(dataloaders['eval'])

    if 'test' in args.phases:
        viewer = VideoViewer(args.video, detector)
        viewer.run()

    if 'export' in args.phases:
        onnx_exporter.export(detector.model, cfg.input_size, 'exported/model.onnx')

    if 'export-mo' in args.phases:
        mo_exporter.export(detector.model, cfg, 'model', folder='exported', postprocess=mo_add_output.add_output)

    if 'export-torch' in args.phases:
        jit_exporter.export(detector, cfg.input_size, 'exported/model.pt')


if __name__ == '__main__':
    parser = helpers.get_default_argparser()
    parser.add_argument('--phases', nargs='+', default=['train', 'eval'],
                        choices=['train', 'eval', 'test', 'export', 'export-mo', 'export-torch', 'embed'],
                        help='One or multiple runtime phases')
    parser.add_argument('--video', type=str,
                        help='Video or a folder (which will be searched recursively) for `test` phase')
    parser.add_argument('--tensorboard', default=False, action='store_true',
                        help='Log to tensorboard')
    args = parser.parse_args()

    helpers.launch(args, main)
