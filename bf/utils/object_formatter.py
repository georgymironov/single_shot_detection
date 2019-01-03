import os
import string

from .misc_utils import try_int, try_eval


class ObjectFormatter(object):
    def __init__(self, obj):
        self.context = dict()
        self.obj = obj
        self.update_context(dict(os.environ))
        self.update_context(vars(obj))

    def update_context(self, ctx):
        self.context.update(ctx)
        self.format_obj()

    def _format_str(self, attr):
        parsed = [x[1] for x in string.Formatter().parse(attr)]
        if all(x is None for x in parsed):
            return attr
        fields = {}
        for field in parsed:
            if field is None:
                continue
            value = self.context.get(field)
            if value is not None:
                fields[field] = value
            else:
                raise ValueError(f'{field} field missing in context')
        return try_int(try_eval(attr.format(**fields)))

    def _format_dict(self, dict_):
        for k, v in dict_.items():
            if isinstance(v, str):
                dict_[k] = self._format_str(v)
            if isinstance(v, dict):
                dict_[k] = self._format_dict(v)
            if isinstance(v, list):
                dict_[k] = self._format_list(v)
        return dict_

    def _format_list(self, list_):
        for i, x in enumerate(list_):
            if isinstance(x, str):
                list_[i] = self._format_str(x)
            if isinstance(x, dict):
                list_[i] = self._format_dict(x)
            if isinstance(x, list):
                list_[i] = self._format_list(x)
        return list_

    def format_obj(self):
        obj = self.obj
        for attr_name in dir(obj):
            if attr_name.startswith('__'):
                continue
            attr = getattr(obj, attr_name)
            if isinstance(attr, str):
                setattr(obj, attr_name, self._format_str(attr))
            if isinstance(attr, dict):
                setattr(obj, attr_name, self._format_dict(attr))
            if isinstance(attr, list):
                setattr(obj, attr_name, self._format_list(attr))
