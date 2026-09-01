"""Privacy-safe local latency instrumentation for one Chainlit turn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from time import perf_counter
from typing import Any
from uuid import uuid4

from agents import RunHooks


PERFORMANCE_LOGGING_ENV_VAR = "ML_AGENT_PERF_LOGGING"


def performance_logging_enabled() -> bool:
    """Return whether local timing lines should be printed to the terminal."""

    value = os.getenv(PERFORMANCE_LOGGING_ENV_VAR, "1").strip().casefold()
    return value not in {"0", "false", "no", "off"}


@dataclass
class ToolTiming:
    """One local tool invocation without arguments or returned content."""

    key: str
    name: str
    started_at: float
    ended_at: float | None = None
    status: str = "running"

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at


@dataclass
class TurnPerformance:
    """Collect and print monotonic timings for a single user message."""

    enabled: bool = field(default_factory=performance_logging_enabled)
    turn_id: str = field(default_factory=lambda: uuid4().hex[:8])
    started_at: float = field(default_factory=perf_counter)
    first_model_event_at: float | None = None
    first_visible_token_at: float | None = None
    attachment_processed: bool = False
    native_file_input_included: bool = False
    model_turn_count: int = 0
    direct_tool_output: bool = False
    second_model_synthesis: bool = False
    tool_timings: list[ToolTiming] = field(default_factory=list)
    _active_tools: dict[str, ToolTiming] = field(default_factory=dict, repr=False)
    _stage_starts: dict[str, float] = field(default_factory=dict, repr=False)
    _tool_sequence: int = 0
    _finished: bool = False

    def mark(self, event: str, **fields: str | int | float | bool) -> None:
        """Print one compact event relative to the turn start."""

        if not self.enabled:
            return
        details = " ".join(
            f"{key}={_format_field(value)}" for key, value in fields.items()
        )
        suffix = f" {details}" if details else ""
        print(f"[PERF {self.turn_id}] event={event} t={self.elapsed_seconds():.3f}s{suffix}")

    def elapsed_seconds(self, timestamp: float | None = None) -> float:
        return (timestamp if timestamp is not None else perf_counter()) - self.started_at

    def set_attachment_context(
        self,
        *,
        attachment_processed: bool,
        native_file_input_included: bool,
    ) -> None:
        self.attachment_processed = attachment_processed
        self.native_file_input_included = native_file_input_included

    def stage_started(
        self,
        stage_key: str,
        event: str,
        **fields: str | int | float | bool,
    ) -> None:
        self._stage_starts[stage_key] = perf_counter()
        self.mark(event, **fields)

    def stage_ended(
        self,
        stage_key: str,
        event: str,
        **fields: str | int | float | bool,
    ) -> None:
        started_at = self._stage_starts.pop(stage_key, None)
        duration = perf_counter() - started_at if started_at is not None else 0.0
        self.mark(event, duration_seconds=duration, **fields)

    def mark_first_model_event(self) -> None:
        if self.first_model_event_at is not None:
            return
        self.first_model_event_at = perf_counter()
        self.mark("first_streamed_model_event")

    def mark_first_visible_token(self) -> None:
        if self.first_visible_token_at is not None:
            return
        self.first_visible_token_at = perf_counter()
        self.mark("first_visible_token_sent")

    def model_started(self) -> None:
        self.model_turn_count += 1
        if self.model_turn_count > 1 and (self.tool_timings or self._active_tools):
            if not self.second_model_synthesis:
                self.second_model_synthesis = True
                self.mark("second_model_synthesis", enabled=True)
        self.mark("model_turn_start", model_turn=self.model_turn_count)

    def model_ended(self) -> None:
        self.mark("model_turn_end", model_turn=self.model_turn_count)

    def tool_started(self, tool_name: str, tool_call_id: str | None) -> None:
        self._tool_sequence += 1
        key = tool_call_id or f"{tool_name}:{self._tool_sequence}"
        timing = ToolTiming(key=key, name=tool_name, started_at=perf_counter())
        self._active_tools[key] = timing
        self.mark("tool_start", tool=tool_name, tool_turn=self._tool_sequence)

    def tool_ended(
        self,
        tool_name: str,
        tool_call_id: str | None,
        *,
        reported_failure: bool,
    ) -> None:
        timing = self._pop_active_tool(tool_name, tool_call_id)
        if timing is None:
            return
        timing.ended_at = perf_counter()
        timing.status = "reported_failure" if reported_failure else "ok"
        self.tool_timings.append(timing)
        self.mark(
            "tool_end",
            tool=timing.name,
            duration_seconds=timing.duration_seconds or 0.0,
            status=timing.status,
        )

    def mark_direct_tool_output(self) -> None:
        """Record that a deterministic tool result ended the agent loop."""

        if self.direct_tool_output:
            return
        self.direct_tool_output = True
        self.mark("direct_tool_output", enabled=True)

    def finish(self, *, failed: bool = False) -> None:
        """Close incomplete records and print one turn summary exactly once."""

        if self._finished:
            return
        self._finished = True

        now = perf_counter()
        for timing in list(self._active_tools.values()):
            timing.ended_at = now
            timing.status = "failed"
            self.tool_timings.append(timing)
            self.mark(
                "tool_end",
                tool=timing.name,
                duration_seconds=timing.duration_seconds or 0.0,
                status=timing.status,
            )
        self._active_tools.clear()

        if not self.enabled:
            return

        tool_details = ",".join(
            f"{timing.name}:{(timing.duration_seconds or 0.0):.3f}s:{timing.status}"
            for timing in self.tool_timings
        ) or "none"
        tool_total = sum(timing.duration_seconds or 0.0 for timing in self.tool_timings)
        self.second_model_synthesis = self.second_model_synthesis or (
            bool(self.tool_timings) and self.model_turn_count > 1
        )
        first_event = _format_optional_duration(self.first_model_event_at, self.started_at)
        first_token = _format_optional_duration(self.first_visible_token_at, self.started_at)
        print(
            f"[PERF {self.turn_id}] summary "
            f"total={now - self.started_at:.3f}s "
            f"first_model_event={first_event} "
            f"first_visible_token={first_token} "
            f"model_turns={self.model_turn_count} "
            f"tool_turns={len(self.tool_timings)} "
            f"direct_tool_output={str(self.direct_tool_output).lower()} "
            f"second_model_synthesis={str(self.second_model_synthesis).lower()} "
            f"tool_total={tool_total:.3f}s "
            f"tools={tool_details} "
            f"attachment_processed={str(self.attachment_processed).lower()} "
            f"native_file_input={str(self.native_file_input_included).lower()} "
            f"status={'failed' if failed else 'ok'}"
        )

    def _pop_active_tool(
        self,
        tool_name: str,
        tool_call_id: str | None,
    ) -> ToolTiming | None:
        if tool_call_id and tool_call_id in self._active_tools:
            return self._active_tools.pop(tool_call_id)
        for key, timing in reversed(list(self._active_tools.items())):
            if timing.name == tool_name:
                return self._active_tools.pop(key)
        return None


class PerformanceRunHooks(RunHooks[Any]):
    """Agents SDK lifecycle hooks backed by a single turn tracker."""

    def __init__(self, performance: TurnPerformance) -> None:
        self.performance = performance

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        self.performance.model_started()

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self.performance.model_ended()

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        self.performance.tool_started(
            _tool_name(tool),
            _tool_call_id(context),
        )

    async def on_tool_end(
        self,
        context: Any,
        agent: Any,
        tool: Any,
        result: object,
    ) -> None:
        reported_failure = isinstance(result, Mapping) and result.get("ok") is False
        self.performance.tool_ended(
            _tool_name(tool),
            _tool_call_id(context),
            reported_failure=reported_failure,
        )


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) and name else "unknown_tool"


def _tool_call_id(context: Any) -> str | None:
    value = getattr(context, "tool_call_id", None)
    return value if isinstance(value, str) and value else None


def _format_field(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_optional_duration(timestamp: float | None, started_at: float) -> str:
    if timestamp is None:
        return "n/a"
    return f"{timestamp - started_at:.3f}s"
