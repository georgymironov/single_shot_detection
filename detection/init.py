import functools
import logging

import torch

from bf.builders import base_builder
import detection.sampler
from detection.box_coder import BoxCoder
from detection.detector_builder import DetectorBuilder
from detection.detector_wrapper import DetectorWrapper
from detection.losses.mutibox_loss import MultiboxLoss
from detection.postprocessor import Postprocessor
from detection.target_assigner import TargetAssigner


def init(device,
         model_params,
         box_coder_params,
         postprocess_params,
         loss_params,
         sampler_params,
         target_assigner_params,
         state={},
         preprocess=None,
         resize=None):
    if 'model' in state:
        logging.info('===> Restoring model from checkpoint')
        detector = state['model']
    elif 'model' in model_params['detector']:
        logging.info(f'===> Restoring model from file {model_params["detector"]["model"]}')
        detector = torch.load(model_params['detector']['model'])
    else:
        base = base_builder.create_base(model_params)
        kwargs = {k: v for k, v in model_params['detector'].items() if k in DetectorBuilder.__init__.__code__.co_varnames}
        detector = DetectorBuilder(base, anchor_generator_params=model_params['anchor_generator'], **kwargs).build().to(device)

        if 'weight' in model_params['detector']:
            logging.info(f'===> Loading model weights from file {model_params["detector"]["weight"]}')
            state_dict = detector.state_dict()
            state_dict.update(torch.load(model_params['detector']['weight']))
            detector.load_state_dict(state_dict)

        if 'model_dict' in state:
            logging.info('===> Loading model weights from checkpoint')
            detector.load_state_dict(state['model_dict'])

    sampler = getattr(detection.sampler, sampler_params['name'])
    kwargs = {k: v for k, v in sampler_params.items() if k in sampler.__code__.co_varnames}
    sampler = functools.partial(sampler, **kwargs)

    criterion = MultiboxLoss(sampler=sampler, **loss_params)
    box_coder = BoxCoder(**box_coder_params)
    postprocessor = Postprocessor(box_coder, **postprocess_params)
    target_assigner = TargetAssigner(box_coder, **target_assigner_params)

    detector_wrapper = DetectorWrapper(detector, postprocessor, preprocess, resize)

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

        target_classes, target_locs = target_assigner.encode_ground_truth(ground_truth, priors)
        target = target_classes.to(device), target_locs.to(device)

        loss, class_loss, loc_loss = criterion(prediction, target)

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
