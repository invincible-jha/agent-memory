"""User confirmation resolution strategy.

:class:`UserConfirmationStrategy` defers the contradiction resolution
decision to a human user rather than making an automated choice.  This
is the most conservative strategy — no data is discarded, and both
entries are preserved pending human review.

When this strategy is applied, the :class:`ResolutionResult` will have:

- ``action = "defer"``
- ``resolved_content = None``  (no automated content selected)
- ``requires_user_input = True``
- ``explanation`` contains a prompt that could be surfaced to the user

Usage
-----
>>> from agent_memory.contradiction.models import MemoryEntry
>>> from agent_memory.contradiction.strategies.user_confirmation import UserConfirmationStrategy
>>> existing = MemoryEntry(content="Alice lives in London.")
>>> new = MemoryEntry(content="Alice lives in Berlin.")
>>> strategy = UserConfirmationStrategy()
>>> result = strategy.resolve(existing, new)
>>> result.action
'defer'
>>> result.requires_user_input
True
>>> result.resolved_content is None
True
"""
from __future__ import annotations

from agent_memory.contradiction.models import MemoryEntry
from agent_memory.contradiction.resolution import ResolutionResult, ResolutionStrategyBase


class UserConfirmationStrategy(ResolutionStrategyBase):
    """Defer contradiction resolution to a human user.

    This strategy produces a :class:`ResolutionResult` with
    ``action="defer"`` and ``requires_user_input=True``.  No content
    decision is made automatically — the caller is responsible for
    presenting the conflict to the user and applying their choice.

    Parameters
    ----------
    prompt_template:
        A format string used to produce the ``explanation`` field.
        Receives ``{existing}`` and ``{new}`` as the contents of the
        two conflicting entries.  Defaults to a sensible prompt.

    Example
    -------
    >>> strategy = UserConfirmationStrategy()
    >>> strategy.name
    'user_confirmation'
    """

    _DEFAULT_PROMPT_TEMPLATE: str = (
        "Contradiction detected. Please choose the correct version:\n"
        "  [A] {existing}\n"
        "  [B] {new}\n"
        "Enter A or B to resolve."
    )

    def __init__(self, prompt_template: str | None = None) -> None:
        self._prompt_template = (
            prompt_template
            if prompt_template is not None
            else self._DEFAULT_PROMPT_TEMPLATE
        )

    @property
    def name(self) -> str:
        return "user_confirmation"

    def resolve(self, existing: MemoryEntry, new: MemoryEntry) -> ResolutionResult:
        """Defer the resolution decision to the user.

        Parameters
        ----------
        existing:
            The entry already in memory.
        new:
            The incoming entry that contradicts *existing*.

        Returns
        -------
        ResolutionResult
            Result with ``action="defer"``, ``resolved_content=None``,
            and ``requires_user_input=True``.
        """
        explanation = self._prompt_template.format(
            existing=_truncate(existing.content, 200),
            new=_truncate(new.content, 200),
        )
        return ResolutionResult(
            strategy_used=self.name,
            action="defer",
            resolved_content=None,
            requires_user_input=True,
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_length: int) -> str:
    """Truncate *text* to *max_length* characters, appending ``"..."`` if cut."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
