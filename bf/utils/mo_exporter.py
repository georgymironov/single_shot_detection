import logging
import os
import tempfile

from bf.utils import onnx_exporter, mo_extensions


class _argv_wrapper(object):
    def __init__(self, dict_):
        self.__dict__ = dict_

    def __getattr__(self, attr):
        return None

def export(model, config, filename, folder=None, postprocess=None):
    _, tmp = tempfile.mkstemp()

    onnx_exporter.export(model, config.input_size, tmp)

    from mo.main import driver
    from mo.utils import import_extensions
    from mo.utils.cli_parser import get_absolute_path

    folder = folder or get_absolute_path('.')

    argv = _argv_wrapper({
        'input_model': tmp,
        'framework': 'onnx',
        'model_name': filename,
        'output_dir': folder,
        'log_level': 'ERROR',
        'mean_values': (),
        'scale_values': (),
        'reverse_input_channels': False,
        'data_type': 'float',
        'disable_fusing': False,
        'disable_resnet_optimization': False,
        'disable_gfusing': False,
        'move_to_preprocess': False,
        'extensions': ','.join([import_extensions.default_path(), os.path.dirname(mo_extensions.__file__)]),
        'silent': True
    })
    logging.info('===> Running model optimizer...')
    driver(argv)

    if postprocess:
        postprocess(os.path.join(folder, filename + '.xml'), config)
