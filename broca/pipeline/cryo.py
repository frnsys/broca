import os
import json
import inspect
from hashlib import md5
from sklearn.externals import joblib

base = '/tmp/cryo/'
if not os.path.exists(base):
    os.makedirs(base)


class Cryo():
    def __init__(self, refresh=False):
        self.refresh = refresh

    def __call__(self, func, *args, **kwargs):
        # Compute the signature
        mod = inspect.getmodule(func)
        mod = mod.__name__

        try:
            name = func.__name__
            src = inspect.getsource(func)

        except AttributeError:
            # Get the repr of the Pipe,
            # which reflects its init args
            name = str(func)
            src = inspect.getsource(func.__call__)

        meta = {
            'mod': mod,
            'name': name,
            'args': args,
            'kwargs': kwargs,

            # To see if source code changed
            'src': src
        }
        meta = json.dumps(meta, sort_keys=True).encode('utf-8')
        sig = md5(meta).hexdigest()

        dir = os.path.join(base, mod.replace('.', '/'), name)
        path = os.path.join(dir, sig) + '.pkl'

        # Thaw
        if os.path.exists(path) and not self.refresh:
            result = joblib.load(path)

        else:
            # Compute & freeze
            result = func(*args, **kwargs)

            # Note: this only invalidates subsequent steps
            # if the output changes, which will automatically cause
            # the next step to recompute (since its input changes)

            # Freeze
            if not os.path.exists(dir):
                os.makedirs(dir)
            joblib.dump(result, path)

        return result
