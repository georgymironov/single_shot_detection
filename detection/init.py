import functools
import logging

import torch

from bf.builders import base_builder
from bf.utils.misc_utils import filter_kwargs

import detection.sampler
from detection import detector_builder
from detection.box_coder import BoxCoder
from detection.detector_wrapper import DetectorWrapper
from detection.losses.multibox_loss import MultiboxLoss
from detection.postprocessor import Postprocessor
from detection.target_assigner import TargetAssigner
from detection.utils import model_fixer


def init(device,
         model_args,
         box_coder_args,
         postprocess_args,
         loss_args,
         sampler_args,
         target_assigner_args,
         state={},
         preprocess=None,
         parallel=False,
         distributed=False):
    assert not (parallel and distributed)

    if 'model' in state:
        logging.info('===> Restoring model from checkpoint')
        detector = state['model']
        del state['model']
    elif 'model' in model_args['detector']:
        logging.info(f'===> Restoring model from file {model_args["detector"]["model"]}')
        loaded = torch.load(model_args['detector']['model'], map_location='cpu')

        if type(loaded) is dict and 'model' in loaded:
            detector = loaded['model']
            del loaded['model']
            del loaded
        else:
            detector = loaded
    else:
        base = base_builder.create_base(**model_args['base'])
        detector = filter_kwargs(detector_builder.build)(base,
                                                         anchor_generator_params=model_args['anchor_generator'],
                                                         **model_args['detector'])

        if 'weight' in model_args['detector']:
            logging.info(f'===> Loading model weights from file {model_args["detector"]["weight"]}')
            loaded = torch.load(model_args['detector']['weight'], map_location='cpu')

            if type(loaded) is dict and 'model_dict' in loaded:
                model_dict = loaded['model_dict']
                del loaded['model']
                del loaded
            else:
                model_dict = loaded

            state_dict = detector.state_dict()
            state_dict.update(model_dict)
            state_dict = model_fixer.fix_weights(state_dict)
            detector.load_state_dict(state_dict)

        if 'model_dict' in state:
            logging.info('===> Loading model weights from checkpoint')
            model_dict = model_fixer.fix_weights(state['model_dict'])
            detector.load_state_dict(model_dict)
            del state['model_dict']

    detector = detector.to(device)
    torch.cuda.empty_cache()

    if parallel:
        detector.predictor = torch.nn.parallel.DataParallel(detector.predictor)

    if distributed:
        try:
            from apex.parallel import DistributedDataParallel, convert_syncbn_model
        except ImportError:
            raise ImportError('Distributed training requires apex package installed.')
        detector = convert_syncbn_model(detector)
        detector.predictor = DistributedDataParallel(detector.predictor)

    logging.info(detector)

    sampler = getattr(detection.sampler, sampler_args['name'])
    kwargs = {k: v for k, v in sampler_args.items() if k in sampler.__code__.co_varnames}
    sampler = functools.partial(sampler, **kwargs)

    box_coder = BoxCoder(**box_coder_args)
    criterion = MultiboxLoss(sampler=sampler, box_coder=box_coder, **loss_args)
    postprocessor = Postprocessor(box_coder, **postprocess_args)
    target_assigner = TargetAssigner(**target_assigner_args)

    detector_wrapper = DetectorWrapper(detector, preprocess, postprocessor)

    def init_epoch_state():
        return {
            'class_loss': 0.0,
            'loc_loss': 0.0,
            'loss': 0.0
        }

    def step_fn(step, phase, batch, state):
        imgs, ground_truth = batch
        imgs = imgs.to(device)

        *prediction, priors = detector(imgs)

        target = target_assigner.encode_ground_truth(ground_truth, priors)
        target = target.to(device)

        loss, class_loss, loc_loss = criterion(prediction, priors.to(device), target)

        prediction = [x.detach() for x in prediction]

        if phase == 'eval':
            prediction = postprocessor.postprocess(prediction, priors)

        if step == 0:
            [setattr(step_fn, attr, 0.0) for attr in ['class_loss', 'loc_loss', 'loss']]

        step_fn.class_loss += class_loss.item()
        step_fn.loc_loss += loc_loss.item()
        step_fn.loss = step_fn.class_loss + step_fn.loc_loss

        state['class_loss'] = step_fn.class_loss / (step + 1)
        state['loc_loss'] = step_fn.loc_loss / (step + 1)
        state['loss'] = step_fn.loss / (step + 1)

        return loss, prediction, state

    return detector_wrapper, init_epoch_state, step_fn
