import logging
import time

import torch

import bf.eval
from bf.utils.event_emitter import EventEmitter


class Trainer(EventEmitter):
    def __init__(self,
                 epochs,
                 phases,
                 model,
                 init_epoch_state_fn=None,
                 step_fn=None,
                 accumulation_steps=1,
                 metrics={},
                 eval_every=1):
        super(Trainer, self).__init__()

        self.epochs = epochs
        self.phases = phases
        self.model = model
        self.init_epoch_state_fn = init_epoch_state_fn
        self.step_fn = step_fn
        self.accumulation_steps = accumulation_steps
        self.eval_every = eval_every

        self.state = {
            'epoch': 0,
            'global_step': -1,
            'model': model
        }

        self.evaluator = bf.eval.Evaluator(model,
                                           init_epoch_state_fn,
                                           step_fn,
                                           metrics)

    @property
    def global_step(self):
        return self.state['global_step']

    def resume(self, initial_epoch, initial_step):
        self.state['epoch'] = initial_epoch
        self.state['global_step'] = initial_step
        logging.info(f'>> Resuming from: step: {initial_epoch}, epoch: {initial_epoch}')

    def _train_epoch(self, dataloader, num_batches=None):
        start = time.time()
        num_batches = num_batches or len(dataloader)
        epoch_state = self.init_epoch_state_fn()

        self.model.train()

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break

            if (step + 1) % self.accumulation_steps == 0:
                self.state['global_step'] += 1
                self.emit('step_start', phase='train', global_state=self.state, state=epoch_state)

            with torch.enable_grad():
                loss, _, epoch_state = self.step_fn(step, 'train', batch, epoch_state)
                loss.backward()

            if (step + 1) % self.accumulation_steps == 0:
                self.emit('step_end', phase='train', global_state=self.state, state=epoch_state)

        elapsed = time.time() - start
        logging.info(f'\n[train] finished in {elapsed//60:.0f}m {elapsed%60:.0f}s')

        return epoch_state

    def _get_phase_len(self, dataloaders, num_batches_per_epoch=None):
        phase_len = {'eval': len(dataloaders['eval'])}
        phase_len['train'] = num_batches_per_epoch or len(dataloaders['train'])
        phase_len['train'] //= self.accumulation_steps
        return phase_len

    def run(self, dataloaders, num_batches_per_epoch=None):
        start = time.time()
        phase_len = self._get_phase_len(dataloaders, num_batches_per_epoch)

        self.emit('start', global_state=self.state)

        for epoch in range(self.state['epoch'], self.epochs):
            logging.info(f'Epoch: {epoch}/{self.epochs-1}')

            self.state['epoch'] = epoch
            epoch_state = {}

            self.emit('epoch_start', global_state=self.state)

            for phase in self.phases:
                if phase == 'eval' and (epoch + 1) % self.eval_every != 0:
                    continue

                self.emit('phase_start', phase=phase, global_state=self.state, phase_len=phase_len[phase])

                if phase == 'train':
                    phase_state = self._train_epoch(dataloaders['train'], num_batches=num_batches_per_epoch)
                if phase == 'eval':
                    phase_state = self.evaluator.run(dataloaders['eval'])

                for k, v in phase_state.items():
                    epoch_state[f'{phase}_{k}'] = v

                if phase == 'train':
                    self.state['model_dict'] = self.model.state_dict()

                self.emit('phase_end', phase=phase, global_state=self.state, phase_state=phase_state)

            self.emit('epoch_end', phase=phase, global_state=self.state, epoch_state=epoch_state)

        elapsed = time.time() - start
        logging.info(f'Training finished. Total time spent: {elapsed//60:.0f}m {elapsed%60:.0f}s')
