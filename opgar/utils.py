import tqdm

import functools


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm
    https://github.com/tqdm/tqdm/issues/313
    """

    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.tqdm.write(x, file=self.file, end="")

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


class Utils:
    
    @staticmethod
    def flatten_nested_dict(parent):
        # Flatten period_result from dict-of-dicts to single-level dict
        #   e.g. {"a1": {"a2": []}, "b1"={"b2":[]}} -> {('a1', 'a2'): [], ('b1', 'b2'): []}
        #   This makes it easier to convert it to a MultiIndex Series/DataFrame later when concatenating results over
        #   multiple time-steps.
        return {
            (outKey, inKey): val
            for outKey, inDict in parent.items()
            for inKey, val in inDict.items()
        }
        
    @staticmethod
    def strategy_to_latex(strategy, include_nnn=False):
        """Takes in a string in the form XXX_YYY and converts it to a latex compatible string for formatting."""
        root, subscript = strategy.split("_")
        new_root = Utils.new_strategy_name_mapping[root]
        if subscript == "NNN" and not include_nnn:
            return new_root
        else:
            return "%s$_{%s}$" % (new_root, subscript)
