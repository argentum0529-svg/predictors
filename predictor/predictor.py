"""Base class inherited by every predictor. Not tied to one method."""

from __future__ import annotations

from typing import ClassVar


class BasePredictor:
    """Registry of predictors by the name used on the command line."""

    name: str = ""
    _registry: ClassVar[dict[str, type[BasePredictor]]] = {}

    def predict(self, instrument: str) -> dict:
        raise NotImplementedError

    def description(self) -> str:
        return ""

    @classmethod
    def register(cls, name: str, predictor_cls: type[BasePredictor]) -> None:
        cls._registry[name.strip().lower()] = predictor_cls

    @classmethod
    def get(cls, name: str) -> BasePredictor:
        key = name.strip().lower()
        predictor_cls = cls._registry.get(key)
        if predictor_cls is None:
            known = ", ".join(sorted(cls._registry)) or "(none)"
            raise KeyError(f"no predictor {name!r}; known: {known}")
        return predictor_cls()
