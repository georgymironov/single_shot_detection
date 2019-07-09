from bf.utils.object_formatter import ObjectFormatter


class ConfigWrapper(object):
    def __init__(self, config):
        self.config = config
        self.formatter = ObjectFormatter(config)

    def update(self, ctx):
        self.formatter.update_context(ctx)

    def __getattr__(self, name):
        return getattr(self.config, name, {})

    def is_voc(self, phase):
        return self.config.dataset.get(phase, {}).get('name', None) == 'Voc'

    def set_phases(self, phases):
        self.phases = phases
        for phase in ['train', 'eval']:
            if phase not in self.phases and phase in self.config.dataset:
                del self.config.dataset[phase]
