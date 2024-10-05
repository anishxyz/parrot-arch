import contextvars
from functools import wraps
from typing import Any, Dict, Callable

StateType = Dict[str, Any]


class Tasker:
    def __init__(self):
        self._context = contextvars.ContextVar("_context", default=[])
        self._history = contextvars.ContextVar("_history", default=[])
        self._state = contextvars.ContextVar("_state", default={})

    def __call__(self, cls=None, **kwargs):
        if cls is not None and isinstance(cls, type):
            # Decorator used without arguments
            self._init_tasker_class(cls)
            return cls
        else:
            # Decorator used with arguments
            def wrapper(cls):
                self._init_tasker_class(cls, **kwargs)
                return cls

            return wrapper

    def _init_tasker_class(self, cls, **kwargs):
        cls._context = self._context
        cls._history = self._history
        cls._state = self._state

        memory = kwargs.get('memory', False)  # Default to False if not provided
        if memory:
            cls.memory = pd.DataFrame()  # Initialize with an empty DataFrame
        else:
            cls.memory = None  # Set to None if memory is False

    def setup(self, method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "_state"):
                raise TypeError(
                    "@tasker.setup can only be used in a @tasker decorated class"
                )

            current_state = self._state.get()
            new_state = method(self, *args, **kwargs)

            current_state.update(new_state)
            self._state.set(current_state)

            return new_state

        return wrapper

    def run(self, method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_state'):
                self._state = {}
            return method(self, *args, **kwargs)

        return wrapper


tasker = Tasker()
