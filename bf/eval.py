import time

import torch


class Evaluator(object):
    def __init__(self,
                 model,
                 init_epoch_state_fn=None,
                 step_fn=None,
                 metrics={}):

        self.model = model
        self.init_epoch_state_fn = init_epoch_state_fn
        self.step_fn = step_fn
        self.metrics = metrics

    def run(self, dataloader):
        start = time.time()
        epoch_len = len(dataloader)
        state = self.init_epoch_state_fn()

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

            messages = [f'[eval] step: {step}/{epoch_len-1}'] + [f'{k}: {v:6f}' for k, v in state.items()]
            print(f'{", ".join(messages)}    ', end='\r')

        elapsed = time.time() - start
        print(f'\n[eval] finished in {elapsed//60:.0f}m {elapsed%60:.0f}s')

        predictions = torch.cat(predictions, dim=0)

        for name, metric in self.metrics.items():
            state[name] = metric(predictions, ground_truths)

        return state
