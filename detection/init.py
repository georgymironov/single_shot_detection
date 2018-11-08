from bf.builders import base_builder
from detection.box_coder import BoxCoder
from detection.detector_builder import DetectorBuilder
from detection.losses.mutibox_loss import MultiboxLoss
from detection.postprocessor import Postprocessor
from detection.target_assigner import TargetAssigner


def init(device,
         num_classes,
         model_params,
         box_coder_params,
         postprocess_params,
         loss_params,
         target_assigner_params,
         state={}):
    base = base_builder.create_base(model_params)

    kwargs = {k: v for k, v in model_params['detector'].items() if k in DetectorBuilder.__init__.__code__.co_varnames}
    detector = DetectorBuilder(base, num_classes=num_classes, **kwargs).create().to(device)

    if 'model' in state:
        print('===> Loading model weights from checkpoint')
        detector.load_state_dict(state['model'])

    criterion = MultiboxLoss(**loss_params)
    box_coder = BoxCoder(**box_coder_params)
    postprocessor = Postprocessor(box_coder, **postprocess_params)
    target_assigner = TargetAssigner(box_coder, **target_assigner_params)

    def init_epoch_state():
        return {
            'class_loss': 0.0,
            'loc_loss': 0.0,
            'loss': 0.0
        }

    def step_fn(step, phase, batch, state):
        device = next(detector.parameters()).device
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

    return detector, init_epoch_state, step_fn
