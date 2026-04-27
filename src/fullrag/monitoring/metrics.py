from __future__ import annotations

from collections import defaultdict


class MetricsRegistry:
    def __init__(self) -> None:
        self.counters: dict[str, int] = defaultdict(int)
        self.histograms: dict[str, list[float]] = defaultdict(list)

    def inc(self, metric: str, value: int = 1) -> None:
        self.counters[metric] += value

    def observe(self, metric: str, value: float) -> None:
        self.histograms[metric].append(value)

    def snapshot(self) -> dict[str, object]:
        return {"counters": dict(self.counters), "histograms": dict(self.histograms)}

    def prometheus_text(self) -> str:
        lines: list[str] = []
        for metric, value in self.counters.items():
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"{metric} {value}")
        for metric, values in self.histograms.items():
            lines.append(f"# TYPE {metric} summary")
            count = len(values)
            total = sum(values)
            lines.append(f"{metric}_count {count}")
            lines.append(f"{metric}_sum {total}")
        return "\n".join(lines) + "\n"
