class _Proxy:
    """A thin wrapper around a tokenizer object used by the trainer.

    The trainer sets ``prog_si`` on the tokenizer during progressive training.
    When wrapping a parent tokenizer we need to forward attribute accesses so
    that the trainer behaves as if it is operating on the real tokenizer.
    """

    def __init__(self, parent):
        object.__setattr__(self, '_parent', parent)
        # progressive training stage index; -1 means disabled
        object.__setattr__(self, 'prog_si', -1)

    def __getattr__(self, name):
        # Delegate attribute reads to the wrapped tokenizer
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name in {'_parent', 'prog_si'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)
    def __delattr__(self, name):
        if name in {'_parent', 'prog_si'}:
            object.__delattr__(self, name)
        else:
            delattr(self._parent, name)

