import inspect


def try_int(x):
    try:
        x = int(x)
    finally:
        return x

def try_float(x):
    try:
        x = float(x)
    finally:
        return x

def try_eval(x):
    try:
        x = eval(x)
    finally:
        return x

def filter_kwargs(func):
    def wrapped_func(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(func).parameters.keys()}
        return func(*args, **kwargs)
    return wrapped_func

def get_ctor(module, name):
    return filter_kwargs(getattr(module, name))
