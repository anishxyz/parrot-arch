import contextvars
from functools import wraps
from typing import Any, Dict, Callable

StateType = Dict[str, Any]


class Tasker:
    def __init__(self):
        self._context = contextvars.ContextVar("_context", default=[])
        self._history = contextvars.ContextVar("_history", default=[])
        self._state = contextvars.ContextVar("_state", default={})

    def __call__(self, cls):
        self._init_tasker_class(cls)
        return cls

    def _init_tasker_class(self, cls):
        cls._context = self._context
        cls._history = self._history
        cls._state = self._state

    def setup(self, setup_func: Callable[..., StateType]):
        def decorator(method):
            @wraps(method)
            def wrapper(self, *args, **kwargs):
                if not hasattr(self, "_state"):
                    raise TypeError(
                        "@tasker.setup can only be used in a @tasker decorated class"
                    )

                current_state = self._state.get()
                new_state = setup_func(self)

                current_state.update(new_state)
                self._state.set(current_state)

                return method(self, *args, **kwargs)

            return wrapper

        return decorator

    def run(self, run_func: Callable):
        def decorator(method):
            @wraps(method)
            def wrapper(self, *args, **kwargs):
                # Ensure that _state is initialized
                if not hasattr(self, '_state'):
                    self._state = {}
                return method(self, *args, **kwargs)

            return wrapper

        return decorator


tasker = Tasker()
