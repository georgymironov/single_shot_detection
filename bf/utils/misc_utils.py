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
        kwargs = {k: v for k, v in kwargs.items() if k in func.__code__.co_varnames}
        return func(*args, **kwargs)
    return wrapped_func
