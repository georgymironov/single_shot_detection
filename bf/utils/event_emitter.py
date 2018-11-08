from collections import defaultdict


class EventEmitter(object):
    def __init__(self):
        self.callbacks = defaultdict(list)

    def add_event_handler(self, event_name, callback):
        self.callbacks[event_name].append(callback)

    def emit(self, event_name, *args, **kwargs):
        for callback in self.callbacks[event_name]:
            callback(*args, **kwargs)

    def on(self, event_name, *args, **kwargs):
        def decorator(func):
            self.add_event_handler(event_name, func)
            return func
        return decorator
