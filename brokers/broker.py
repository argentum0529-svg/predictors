"""Base class inherited by every broker. Not tied to one exchange or vendor."""

from __future__ import annotations

from typing import ClassVar


class BaseBroker:
    """Default broker per exchange. Subclasses register themselves."""

    _defaults: ClassVar[dict[str, type[BaseBroker]]] = {}

    def exchanges(self) -> list[str]:
        return []

    @classmethod
    def register(cls, exchange: str, broker_cls: type[BaseBroker]) -> None:
        cls._defaults[exchange.strip().lower()] = broker_cls

    @classmethod
    def default_broker(cls, exchange: str) -> BaseBroker:
        key = exchange.strip().lower()
        broker_cls = cls._defaults.get(key)
        if broker_cls is None:
            known = ", ".join(sorted(cls._defaults)) or "(none)"
            raise KeyError(f"no default broker for {exchange!r}; known: {known}")
        return broker_cls()
