"""Contradiction resolution orchestrator.

:class:`ContradictionResolver` (E14.2) orchestrates resolution of contradictions
between :class:`~agent_memory.contradiction.models.MemoryEntry` pairs using
interchangeable :class:`ResolutionStrategyBase` implementations.

This is a new, higher-level resolver distinct from the backward-compatible
:class:`~agent_memory.contradiction.resolver.ContradictionResolver` (which works
with Pydantic ``MemoryEntry`` types from ``memory.types``).  This module uses
the lightweight frozen-dataclass ``MemoryEntry`` from ``contradiction.models``.

Built-in strategies (registered by default)
--------------------------------------------
- ``"latest_wins"``      — keep the more recently timestamped entry
- ``"user_confirmation"`` — defer the decision to the user
- ``"merge"``             — merge non-conflicting sentence-level content

Additional strategies can be registered at runtime via
:meth:`ContradictionResolver.register_strategy`.

Usage
-----
>>> from agent_memory.contradiction.models import MemoryEntry
>>> from agent_memory.contradiction.resolution import ContradictionResolver
>>> resolver = ContradictionResolver()
>>> existing = MemoryEntry(content="The capital of France is Paris.")
>>> new = MemoryEntry(content="The capital of France is Lyon.")
>>> result = resolver.resolve(existing, new)
>>> result.strategy_used
'latest_wins'
>>> result.action in ("replace", "merge", "defer", "keep_both")
True
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agent_memory.contradiction.models import MemoryEntry


# ---------------------------------------------------------------------------
# ResolutionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionResult:
    """The outcome of applying a resolution strategy to a contradiction.

    Attributes
    ----------
    strategy_used:
        Name of the strategy that produced this resolution.
    action:
        One of ``"replace"``, ``"merge"``, ``"defer"``, ``"keep_both"``.
    resolved_content:
        The resolved memory content.  ``None`` when ``action="defer"``
        (no automated decision was made).
    requires_user_input:
        ``True`` when human review is needed before the contradiction is
        considered resolved.
    explanation:
        Human-readable rationale for the resolution decision.
    """

    strategy_used: str
    action: str  # "replace", "merge", "defer", "keep_both"
    resolved_content: str | None  # None when deferred
    requires_user_input: bool
    explanation: str


# ---------------------------------------------------------------------------
# ResolutionStrategyBase (ABC)
# ---------------------------------------------------------------------------


class ResolutionStrategyBase(ABC):
    """Abstract base class for contradiction resolution strategies.

    All strategies must implement :meth:`resolve` and expose a :attr:`name`
    property.

    Subclasses
    ----------
    - :class:`~agent_memory.contradiction.strategies.LatestWinsStrategy`
    - :class:`~agent_memory.contradiction.strategies.UserConfirmationStrategy`
    - :class:`~agent_memory.contradiction.strategies.MergeResolutionStrategy`
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine-readable strategy name used for registration."""
        ...

    @abstractmethod
    def resolve(self, existing: MemoryEntry, new: MemoryEntry) -> ResolutionResult:
        """Resolve the contradiction between *existing* and *new*.

        Parameters
        ----------
        existing:
            The memory entry currently stored.
        new:
            The incoming memory entry that contradicts *existing*.

        Returns
        -------
        ResolutionResult
            The resolution outcome.
        """
        ...


# ---------------------------------------------------------------------------
# ContradictionResolver
# ---------------------------------------------------------------------------


class ContradictionResolver:
    """Orchestrates contradiction resolution using pluggable strategies.

    At construction time the built-in strategies (``latest_wins``,
    ``user_confirmation``, ``merge``) are registered automatically.
    Additional strategies can be added via :meth:`register_strategy`.

    Parameters
    ----------
    default_strategy:
        Name of the strategy to use when :meth:`resolve` is called without
        an explicit ``strategy`` override.  Defaults to ``"latest_wins"``.

    Raises
    ------
    ValueError
        If *default_strategy* is not a recognised strategy name at
        construction time.

    Example
    -------
    >>> resolver = ContradictionResolver()
    >>> resolver.default_strategy
    'latest_wins'
    >>> existing = MemoryEntry(content="Sky is blue.")
    >>> new = MemoryEntry(content="Sky is green.")
    >>> result = resolver.resolve(existing, new)
    >>> result.strategy_used
    'latest_wins'
    """

    def __init__(self, default_strategy: str = "latest_wins") -> None:
        # Lazy import to avoid circular dependencies at module load time
        from agent_memory.contradiction.resolution_strategies.latest_wins import LatestWinsStrategy
        from agent_memory.contradiction.resolution_strategies.merge import MergeResolutionStrategy
        from agent_memory.contradiction.resolution_strategies.user_confirmation import UserConfirmationStrategy

        self._registry: dict[str, ResolutionStrategyBase] = {}

        # Register built-in strategies
        for strategy in (
            LatestWinsStrategy(),
            UserConfirmationStrategy(),
            MergeResolutionStrategy(),
        ):
            self._registry[strategy.name] = strategy

        if default_strategy not in self._registry:
            raise ValueError(
                f"Unknown default strategy {default_strategy!r}. "
                f"Valid options: {sorted(self._registry.keys())!r}"
            )
        self._default_strategy = default_strategy

    @property
    def default_strategy(self) -> str:
        """The name of the default resolution strategy."""
        return self._default_strategy

    @property
    def registered_strategies(self) -> list[str]:
        """Names of all currently registered strategies."""
        return sorted(self._registry.keys())

    def register_strategy(
        self,
        name: str,
        strategy: ResolutionStrategyBase,
    ) -> None:
        """Register a custom resolution strategy.

        Parameters
        ----------
        name:
            The strategy name used to look it up at resolution time.
            Overrides any existing strategy with the same name.
        strategy:
            An instance of a :class:`ResolutionStrategyBase` subclass.

        Example
        -------
        >>> class MyStrategy(ResolutionStrategyBase):
        ...     @property
        ...     def name(self): return "my_strategy"
        ...     def resolve(self, existing, new): ...
        >>> resolver = ContradictionResolver()
        >>> resolver.register_strategy("my_strategy", MyStrategy())
        """
        if not isinstance(strategy, ResolutionStrategyBase):
            raise TypeError(
                f"strategy must be a ResolutionStrategyBase instance, "
                f"got {type(strategy).__name__}"
            )
        self._registry[name] = strategy

    def resolve(
        self,
        existing: MemoryEntry,
        new: MemoryEntry,
        strategy: str | None = None,
    ) -> ResolutionResult:
        """Resolve the contradiction between *existing* and *new*.

        Parameters
        ----------
        existing:
            The memory entry currently stored.
        new:
            The incoming memory entry that contradicts *existing*.
        strategy:
            Name of the strategy to use.  Falls back to
            :attr:`default_strategy` when ``None``.

        Returns
        -------
        ResolutionResult
            The resolution outcome from the selected strategy.

        Raises
        ------
        ValueError
            If the requested strategy name is not registered.
        """
        active_name = strategy if strategy is not None else self._default_strategy

        strategy_obj = self._registry.get(active_name)
        if strategy_obj is None:
            raise ValueError(
                f"Unknown strategy {active_name!r}. "
                f"Registered strategies: {sorted(self._registry.keys())!r}"
            )

        return strategy_obj.resolve(existing, new)

    def resolve_batch(
        self,
        pairs: list[tuple[MemoryEntry, MemoryEntry]],
        strategy: str | None = None,
    ) -> list[ResolutionResult]:
        """Resolve multiple contradiction pairs using the same strategy.

        Parameters
        ----------
        pairs:
            List of ``(existing, new)`` memory entry pairs.
        strategy:
            Strategy name override.  Falls back to default when ``None``.

        Returns
        -------
        list[ResolutionResult]
            One result per input pair, in the same order.
        """
        return [
            self.resolve(existing, new, strategy=strategy)
            for existing, new in pairs
        ]
