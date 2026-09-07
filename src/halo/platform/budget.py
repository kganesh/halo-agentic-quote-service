"""Design rule 05: the harness enforces budgets. The prompt does not request them.

Asking a model to be brief is not a limit. This module is a limit. The loop
checks the budget before every step and stops with a reason. A run that goes over
budget produces a clean escalation instead of a truncated answer.
"""

import time
from decimal import Decimal

from pydantic import BaseModel, Field


class Budget(BaseModel):
    """Limits for one agent run. All four are hard stops."""

    wall_clock_seconds: float = Field(gt=0)
    max_tokens: int = Field(gt=0)
    max_tool_calls: int = Field(ge=0)
    max_usd: Decimal = Field(gt=0)


class Usage(BaseModel):
    """What a run has spent so far."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    tool_calls: int = 0
    usd: Decimal = Decimal("0.00")

    @property
    def total_tokens(self) -> int:
        """Every token the run was billed for.

        Cache tokens are counted here because the API reports them separately
        from `input_tokens`, not inside it. Leaving them out would let a cached
        run pass a `max_tokens` limit it had actually exceeded.
        """
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from cache. Zero when nothing is cached.

        The number to watch when turning caching on: if it stays at zero across
        repeated runs, something is invalidating the prefix.
        """
        billed_input = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / billed_input if billed_input else 0.0


class BudgetExceeded(Exception):
    """Records which limit was reached, and whose it was.

    `owner` matters because several budgets are in play at once: each specialist
    holds one, and the run holds another that the shared model client and gateway
    count against. Without it, a run-level trip inside the pricing specialist
    reads as "pricing exhausted its budget", and raising pricing's allowance
    changes nothing — the reason names the wrong thing to fix.
    """

    def __init__(self, dimension: str, limit: object, spent: object, owner: str = "run") -> None:
        super().__init__(f"{owner} budget exceeded on {dimension}: spent {spent}, limit {limit}")
        self.dimension = dimension
        self.owner = owner


class BudgetTracker:
    """Wraps a budget with a clock. One tracker per agent run.

    `owner` names whose allowance this is, so a breach can say which budget to
    raise. It defaults to "run" because that is the one a caller who does not
    care about the distinction is holding.
    """

    def __init__(
        self, budget: Budget, now: callable = time.monotonic, *, owner: str = "run"
    ) -> None:
        self._budget = budget
        self._owner = owner
        self._now = now
        self._started = now()
        self.usage = Usage()

    @property
    def elapsed_seconds(self) -> float:
        return self._now() - self._started

    @property
    def remaining_seconds(self) -> float:
        """Wall clock left before this budget trips. Never negative.

        This is what a caller passes as a request timeout, so that a call which
        would outlast the budget is cut off by the HTTP layer instead of running
        to completion and being noticed afterwards. Zero means the budget is
        already spent; `check` is what turns that into a `BudgetExceeded`.
        """
        return max(0.0, self._budget.wall_clock_seconds - self.elapsed_seconds)

    @property
    def remaining_usd(self) -> Decimal:
        """Dollars left before this budget trips. Never negative.

        `check` runs between steps, so it can only catch an overspend after the
        call that caused it has been paid for. This is the number that lets a
        caller size the next call to fit, which is the only way a per-call
        overshoot is prevented rather than reported.
        """
        return max(Decimal("0.00"), self._budget.max_usd - self.usage.usd)

    def check(self) -> None:
        """Raise if any limit has been passed. Call this before each step."""
        if self.elapsed_seconds > self._budget.wall_clock_seconds:
            raise BudgetExceeded(
                "wall_clock_seconds",
                self._budget.wall_clock_seconds,
                round(self.elapsed_seconds, 2),
                self._owner,
            )
        if self.usage.total_tokens > self._budget.max_tokens:
            raise BudgetExceeded(
                "max_tokens", self._budget.max_tokens, self.usage.total_tokens, self._owner
            )
        if self.usage.tool_calls > self._budget.max_tool_calls:
            raise BudgetExceeded(
                "max_tool_calls", self._budget.max_tool_calls, self.usage.tool_calls, self._owner
            )
        if self.usage.usd > self._budget.max_usd:
            raise BudgetExceeded("max_usd", self._budget.max_usd, self.usage.usd, self._owner)

    def exceeded(self, dimension: str, spent: object) -> BudgetExceeded:
        """The breach for a limit the caller detected on its own.

        `check` covers what the tracker can see by itself. Two things it cannot:
        sizing a call to the remaining dollars needs a rate card, and a request
        cut off in flight is known to the HTTP layer. Both are found outside,
        and both are the same kind of failure — so they are raised as the same
        exception, carrying this tracker's owner and limit rather than a second
        error type the loop would have to learn.
        """
        limits: dict[str, object] = {
            "wall_clock_seconds": self._budget.wall_clock_seconds,
            "max_tokens": self._budget.max_tokens,
            "max_tool_calls": self._budget.max_tool_calls,
            "max_usd": self._budget.max_usd,
        }
        return BudgetExceeded(dimension, limits[dimension], spent, self._owner)

    def record_model_call(
        self,
        input_tokens: int,
        output_tokens: int,
        usd: Decimal,
        *,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.cache_read_tokens += cache_read_tokens
        self.usage.cache_write_tokens += cache_write_tokens
        self.usage.usd += usd

    def record_tool_call(self) -> None:
        self.usage.tool_calls += 1
