import string

from .misc_utils import try_int


class ConfigFormatter(object):
    def __init__(self, context):
        self.context = context

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
        return try_int(eval(attr.format(**fields)))

    def _format_dict(self, d):
        for k, v in d.items():
            if isinstance(v, str):
                d[k] = self._format_str(v)
            if isinstance(v, dict):
                d[k] = self._format_dict(v)
            if isinstance(v, (list, tuple)):
                d[k] = self._format_list(v)
        return d

    def _format_list(self, l):
        l = list(l)
        for i, x in enumerate(l):
            if isinstance(x, str):
                l[i] = self._format_str(x)
            if isinstance(x, dict):
                l[i] = self._format_dict(x)
            if isinstance(x, (list, tuple)):
                l[i] = self._format_list(x)
        return l

    def format_obj(self, obj):
        for attr_name in dir(obj):
            if attr_name.startswith('__'):
                continue
            attr = getattr(obj, attr_name)
            if isinstance(attr, str):
                setattr(obj, attr_name, self._format_str(attr))
            if isinstance(attr, dict):
                setattr(obj, attr_name, self._format_dict(attr))
            if isinstance(attr, (list, tuple)):
                setattr(obj, attr_name, self._format_list(attr))
