if __package__ or '.' in __name__:
    from . import pyann_low
else:
    import pyann_low

__version__ = pyann_low.cvar.ann_version
