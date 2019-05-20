class _no_target(object):
    def __getattr__(self, *args):
        return None

no_target = _no_target()
