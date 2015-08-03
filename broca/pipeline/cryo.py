import os
import json
import base64
import inspect
import numpy as np
import scipy.sparse as sps
from hashlib import md5
from sklearn.externals import joblib

base = '/tmp/cryo/'
if not os.path.exists(base):
    os.makedirs(base)


class CryoEncoder(json.JSONEncoder):
    """
    For handling JSON serialization of numpy arrays and scipy sparse matrices.
    Adapted from <http://stackoverflow.com/a/24375113>
    """
    def default(self, obj):
        """
        if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded
        """
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(obj.data).decode('utf-8')
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        elif sps.issparse(obj):
            data_b64 = base64.b64encode(obj.data).decode('utf-8')
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape,
                        indices=obj.indices,
                        indptr=obj.indptr)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


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
        meta = json.dumps(meta, sort_keys=True, cls=CryoEncoder).encode('utf-8')
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
