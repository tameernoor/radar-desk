"""GPU cost estimates and the monthly budget check.

An attempt costs its run seconds plus the idle window, times the per-second price of the GPU type.
Every attempt of a job adds to its `cost_estimate_usd`, so retries count toward the budget.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from radar_desk.records import Job, Timings

# Device names as torch reports them, matched in this order (A100 before A10, L40S before L4).
_DEVICE_TYPES = [
    ("H100", "H100"),
    ("A100-SXM4-80GB", "A100-80GB"),
    ("A100 80GB", "A100-80GB"),
    ("A100-80GB", "A100-80GB"),
    ("A100", "A100-40GB"),
    ("A10", "A10"),
    ("L40S", "L40S"),
    ("L4", "L4"),
    ("T4", "T4"),
]


def gpu_type_from_device(device: str | None) -> str | None:
    """Map a device name such as "NVIDIA A10G" to a price-table type, or None if unknown."""
    if not device:
        return None
    upper = device.upper()
    for needle, gpu_type in _DEVICE_TYPES:
        if needle in upper:
            return gpu_type
    return None


def price_per_s(gpu: str | None, settings: Any) -> float:
    """Price of `gpu`, falling back to the first requested type when it is not in the table."""
    table = settings.gpu_prices_usd_per_s
    if gpu in table:
        return table[gpu]
    for requested in settings.gpu_list:
        if requested in table:
            return table[requested]
    return max(table.values())


def estimate(
    timings: Timings | dict | None, gpu: str | None, settings: Any, elapsed_s: float | None = None
) -> float:
    """Cost of one attempt: the larger of total_s and the elapsed time since submit, plus the idle window,
    times the price. Elapsed time covers container boot and Modal's own retry, which total_s misses."""
    total = None
    if isinstance(timings, Timings):
        total = timings.total_s
    elif isinstance(timings, dict):
        total = timings.get("total_s")
    seconds = max([float(v) for v in (total, elapsed_s) if v is not None], default=0.0)
    return (seconds + settings.gpu_scaledown_window_s) * price_per_s(gpu, settings)


def worst_case(settings: Any) -> float:
    """One job at its limit: two attempts at the timeout plus the idle window, on the dearest requested
    type, since Modal may run a fallback."""
    seconds = 2 * settings.gpu_timeout_s + settings.gpu_scaledown_window_s
    price = max((price_per_s(g, settings) for g in settings.gpu_list), default=price_per_s(None, settings))
    return seconds * price


def budget_allows(month_sum: float, settings: Any, in_flight: float = 0.0) -> bool:
    return month_sum + in_flight + worst_case(settings) <= settings.gpu_monthly_budget_usd


def month_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m")


def iso_at(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(value: str) -> float:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC).timestamp()


class CostService:
    def __init__(self, db: Any, settings: Any) -> None:
        self.db = db
        self.settings = settings

    def spend_for_month(self, month: str) -> float:
        """Every attempt with a stored cost in `month`, including a retried job's earlier attempts."""
        return self.db.sum_cost_for_month(month)

    def in_flight(self) -> float:
        """A worst case per submitted job, whose current attempt has no stored cost yet."""
        return worst_case(self.settings) * len(self.db.list_jobs(state="submitted", limit=100_000))

    def budget_allows_now(self, now: float) -> bool:
        return budget_allows(self.spend_for_month(month_of(now)), self.settings, self.in_flight())

    def attempt_cost(self, job: Job, timings: Timings | dict | None, gpu: str | None, now: float) -> float:
        elapsed = now - parse_iso(job.submitted_at) if job.submitted_at else None
        return estimate(timings, gpu or (job.gpu_requested[0] if job.gpu_requested else None),
                        self.settings, elapsed_s=elapsed)

    def gpu_status(self, backend_name: str, now: float) -> dict:
        """The body of GET /gpu/status. `queued` is oldest first with held jobs included; the hourly
        price is that of the GPU in use or first requested, null on the fake backend."""
        month = month_of(now)
        submitted = self.db.list_jobs(state="submitted", limit=100_000)
        queued = self.db.list_jobs(state="queued", limit=100_000)
        in_flight = submitted[-1] if submitted else None
        price = None
        if backend_name != "fake":
            gpu = (in_flight.gpu_used if in_flight else None) or next(iter(self.settings.gpu_list), None)
            price = round(price_per_s(gpu, self.settings) * 3600, 4)
        return {
            "backend": backend_name,
            "gpu_requested": self.settings.gpu_list,
            "in_flight": in_flight.model_dump() if in_flight else None,
            "held": [j.model_dump() for j in reversed(queued) if j.hold_reason],
            "queued": [j.model_dump() for j in reversed(queued)],
            "price_per_hour_usd": price,
            "spend_month_usd": round(self.spend_for_month(month), 6),
            "budget_usd": self.settings.gpu_monthly_budget_usd,
            "month": month,
        }
