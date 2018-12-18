from bf.utils.object_formatter import ObjectFormatter


class ConfigWrapper(object):
    def __init__(self, config):
        self.config = config
        self.formatter = ObjectFormatter(config)

    def update(self, ctx):
        self.formatter.update_context(ctx)

    def __getattr__(self, name):
        return getattr(self.config, name, None)

    def is_voc(self, phase):
        return self.config.dataset.get(phase, {}).get('name', None) == 'Voc'

    def set_phases(self, phases):
        self.phases = phases
        for phase in list(self.config.dataset.keys()):
            if phase not in phases:
                del self.config.dataset[phase]
