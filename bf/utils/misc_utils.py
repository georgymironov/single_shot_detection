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

def filter_ctor_args(Class):
    def wrapper_function(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in Class.__init__.__code__.co_varnames}
        return Class(*args, **kwargs)
    return wrapper_function

def get_ctor(module, name):
    return filter_ctor_args(getattr(module, name))
