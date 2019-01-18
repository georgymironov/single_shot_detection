transform_type = 'no_target'


class set_transform_type(object):
    def __init__(self, new_transform_type):
        global transform_type
        self.prev = new_transform_type
        transform_type = new_transform_type

    def __enter__(self):
        pass

    def __exit__(self, *args):
        global transform_type
        transform_type = self.prev

class no_target(object):
    def __getitem__(self, *args):
        return None
