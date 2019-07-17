import logging
import time

import torch
import torch.distributed as dist

import bf.eval
from bf.training import callbacks
from bf.utils.event_emitter import EventEmitter


class Trainer(object):
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
        }

        if not dist.is_initialized():
            self.state['model'] = model

        self.event_emitter = EventEmitter()

        device = next(model.parameters()).device

        callbacks.progress(self.event_emitter)
        callbacks.to_device(self.event_emitter, device)
        callbacks.loss(self.event_emitter)

        self.evaluator = bf.eval.Evaluator(model,
                                           init_epoch_state_fn,
                                           step_fn,
                                           metrics,
                                           event_emitter=self.event_emitter)

    @property
    def global_step(self):
        return self.state['global_step']

    def resume(self, initial_epoch, initial_step):
        self.state['epoch'] = initial_epoch
        self.state['global_step'] = initial_step
        logging.info(f'>> Resuming from: step: {initial_step}, epoch: {initial_epoch}')

    def _train_epoch(self, dataloader, num_batches=None):
        start = time.time()
        num_batches = num_batches or len(dataloader)
        phase_len = num_batches // self.accumulation_steps
        phase_state = self.init_epoch_state_fn()

        self.event_emitter.emit('phase_start', phase='train', global_state=self.state, phase_len=phase_len)

        self.model.train()

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break

            if (step + 1) % self.accumulation_steps == 0:
                self.state['global_step'] += 1
                self.event_emitter.emit('step_start', phase='train', step=step, batch=batch, global_state=self.state, state=phase_state)

            with torch.enable_grad():
                loss, _, phase_state = self.step_fn(step, 'train', batch.get(), phase_state)

            if (step + 1) % self.accumulation_steps == 0:
                self.event_emitter.emit('step_end', phase='train', step=step, global_state=self.state, state=phase_state, loss=loss)

        self.state['model_dict'] = self.model.state_dict()

        self.event_emitter.emit('phase_end', phase='train', global_state=self.state, phase_state=phase_state)

        elapsed = time.time() - start
        logging.info(f'\n[train] finished in {elapsed//60:.0f}m {elapsed%60:.0f}s')

        return phase_state

    def run(self, dataloaders, num_batches_per_epoch=None):
        start = time.time()

        self.event_emitter.emit('start', global_state=self.state)

        for epoch in range(self.state['epoch'], self.epochs):
            logging.info(f'Epoch: {epoch}/{self.epochs-1}')

            self.state['epoch'] = epoch
            epoch_state = {}

            self.event_emitter.emit('epoch_start', global_state=self.state)

            for phase in self.phases:
                if phase == 'train':
                    phase_state = self._train_epoch(dataloaders['train'], num_batches=num_batches_per_epoch)
                elif phase == 'eval' and (epoch + 1) % self.eval_every == 0:
                    phase_state = self.evaluator.run(dataloaders['eval'])
                else:
                    continue

                for k, v in phase_state.items():
                    epoch_state[f'{phase}_{k}'] = v

            self.event_emitter.emit('epoch_end', phase=phase, global_state=self.state, epoch_state=epoch_state)

        elapsed = time.time() - start
        logging.info(f'Training finished. Total time spent: {elapsed//60:.0f}m {elapsed%60:.0f}s')
