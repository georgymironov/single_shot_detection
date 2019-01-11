import time

import torch

import bf.eval
from bf.utils.event_emitter import EventEmitter


class Trainer(EventEmitter):
    def __init__(self,
                 epochs,
                 phases,
                 model,
                 optimizer,
                 init_epoch_state_fn=None,
                 step_fn=None,
                 accumulation_steps=1,
                 metrics={},
                 eval_every=1):
        super(Trainer, self).__init__()

        self.epochs = epochs
        self.phases = phases
        self.model = model
        self.optimizer = optimizer
        self.init_epoch_state_fn = init_epoch_state_fn
        self.step_fn = step_fn
        self.accumulation_steps = accumulation_steps
        self.eval_every = eval_every

        self.state = {
            'epoch': 0,
            'global_step': 0,
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

    def _train_epoch(self, dataloader, num_batches=None):
        start = time.time()
        global_state = self.state
        num_batches = len(dataloader) if num_batches is None else num_batches
        epoch_len = num_batches // self.accumulation_steps
        epoch_state = self.init_epoch_state_fn()

        self.model.train()

        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break

            if (step + 1) % self.accumulation_steps == 0:
                global_state['global_step'] += 1
                self.emit('step_start', phase='train', global_state=self.state, state=epoch_state)

            with torch.enable_grad():
                loss, _, epoch_state = self.step_fn(step, 'train', batch, epoch_state)
                loss.backward()

            if (step + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                messages = [f'[train] step: {step//self.accumulation_steps}/{epoch_len-1}']
                for k, v in epoch_state.items():
                    messages.append(f'{k}: {v:6f}')
                for i, x in enumerate(self.optimizer.param_groups):
                    messages.append(f'lr {i}: {x["lr"]:.12f}')

                print(f'{", ".join(messages)}    ', end='\r')

                self.emit('step_end', phase='train', global_state=self.state, state=epoch_state)

        elapsed = time.time() - start
        print(f'\n[train] finished in {elapsed//60:.0f}m {elapsed%60:.0f}s')

        return epoch_state

    def run(self, dataloader, num_batches_per_epoch=None):
        start = time.time()

        for epoch in range(self.state['epoch'], self.epochs):
            print(f'Epoch: {epoch}/{self.epochs-1}')

            self.state['epoch'] = epoch
            epoch_state = {}

            self.emit('epoch_start', global_state=self.state)

            for phase in self.phases:
                if phase == 'val' and (epoch + 1) % self.eval_every != 0:
                    continue

                self.emit('phase_start', phase=phase, global_state=self.state)

                if phase == 'train':
                    phase_state = self._train_epoch(dataloader['train'], num_batches=num_batches_per_epoch)
                if phase == 'val':
                    phase_state = self.evaluator.run(dataloader['val'])

                for k, v in phase_state.items():
                    epoch_state[f'{phase}_{k}'] = v

                if phase == 'train':
                    self.state['model'] = self.model.state_dict()
                    self.state['optimizer'] = self.optimizer.state_dict()

                self.emit('phase_end', phase=phase, global_state=self.state, epoch_state=epoch_state, phase_state=phase_state)

            self.emit('epoch_end', phase=phase, global_state=self.state, epoch_state=epoch_state)

        elapsed = time.time() - start
        print(f'Training finished. Total time spent: {elapsed//60:.0f}m {elapsed%60:.0f}s')
