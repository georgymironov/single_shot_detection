import logging
import time

import torch

from bf.utils.event_emitter import EventEmitter


class Evaluator(object):
    def __init__(self,
                 model,
                 init_epoch_state_fn=None,
                 step_fn=None,
                 metrics={},
                 event_emitter=None):

        self.model = model
        self.init_epoch_state_fn = init_epoch_state_fn
        self.step_fn = step_fn
        self.metrics = metrics
        self.event_emitter = event_emitter or EventEmitter()

    def run(self, dataloader):
        start = time.time()
        state = self.init_epoch_state_fn()

        self.event_emitter.emit('phase_start', phase='eval', phase_len=len(dataloader))

        self.model.eval()

        ground_truths = []
        predictions = []

        for step, batch in enumerate(dataloader):
            img, ground_truth = batch

            with torch.no_grad():
                _, prediction, state = self.step_fn(step, 'eval', batch, state)

            ground_truths += list(ground_truth)

            for i, p in enumerate(prediction):
                index = step * dataloader.batch_size + i
                predictions.append(torch.cat([torch.full((p.size(0), 1), index, dtype=torch.float32), p], dim=1))

            self.event_emitter.emit('step_end', phase='eval', step=step, state=state)

        self.event_emitter.emit('phase_end', phase='eval', phase_state=state)

        elapsed = time.time() - start
        logging.info(f'\n[eval] finished in {elapsed//60:.0f}m {elapsed%60:.0f}s')

        predictions = torch.cat(predictions, dim=0)

        for name, metric in self.metrics.items():
            state[name] = metric(predictions, ground_truths)

        return state
