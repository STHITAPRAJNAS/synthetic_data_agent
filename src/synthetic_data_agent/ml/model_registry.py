from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import structlog

from ..config import get_settings
from ..models.quality_report import QualityReport

logger = structlog.get_logger()


class ModelRegistry:
    """Tracks trained model strategy and quality history to drive the adaptive feedback loop.

    History is persisted as JSON to ``storage_path/model_history.json``.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self.storage_path = storage_path or get_settings().model_storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_path = self.storage_path / "model_history.json"
        # Load synchronously only in __init__ (called before the event loop is running)
        self.history: dict[str, list[dict[str, Any]]] = self._load_history_sync()

    # ------------------------------------------------------------------
    # Sync helpers — only safe to call outside a running event loop
    # ------------------------------------------------------------------

    def _load_history_sync(self) -> dict[str, list[dict[str, Any]]]:
        if self.history_path.exists():
            with open(self.history_path) as f:
                return json.load(f)
        return {}

    # ------------------------------------------------------------------
    # Async I/O — called from async agent methods
    # ------------------------------------------------------------------

    async def _save_history(self) -> None:
        """Persist history to disk without blocking the event loop."""
        data = json.dumps(self.history, indent=2)
        await asyncio.to_thread(self.history_path.write_text, data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record_run(
        self,
        table_fqn: str,
        strategy: str,
        quality_report: QualityReport,
    ) -> None:
        """Record the outcome of a generation run for future strategy selection.

        Args:
            table_fqn: Fully-qualified table name.
            strategy: ML strategy used ('ctgan', 'tvae', 'copula', 'timegan').
            quality_report: The QualityReport produced by ValidatorAgent.
        """
        if table_fqn not in self.history:
            self.history[table_fqn] = []

        ks_values = list(quality_report.ks_test_results.values())
        self.history[table_fqn].append({
            "strategy": strategy,
            "overall_pass": quality_report.overall_pass,
            "ks_test_avg": sum(ks_values) / len(ks_values) if ks_values else 0.0,
            "correlation_delta": quality_report.correlation_delta_frobenius,
            "timestamp": quality_report.generated_at.isoformat(),
        })
        await self._save_history()
        logger.info(
            "Recorded model run",
            table=table_fqn,
            strategy=strategy,
            passed=quality_report.overall_pass,
        )

    def best_strategy_for(self, table_fqn: str) -> Optional[str]:
        """Return the historically best-performing strategy for a table, or None on first run.

        Args:
            table_fqn: Fully-qualified table name.

        Returns:
            Strategy string (e.g. 'ctgan') or None if no history exists.
        """
        runs = self.history.get(table_fqn)
        if not runs:
            return None

        # Prefer strategies that passed quality gates; break ties by avg KS score
        passing = [r for r in runs if r.get("overall_pass")]
        candidates = passing if passing else runs

        strategy_scores: dict[str, list[float]] = {}
        for run in candidates:
            strat = run["strategy"]
            score = run.get("ks_test_avg", 0.0)
            strategy_scores.setdefault(strat, []).append(score)

        best = max(
            strategy_scores,
            key=lambda s: sum(strategy_scores[s]) / len(strategy_scores[s]),
        )
        logger.info("Best historical strategy", table=table_fqn, strategy=best)
        return best

    def get_tuning_suggestions(
        self, table_fqn: str, failed_report: QualityReport
    ) -> dict[str, Any]:
        """Derive parameter adjustments from a failed QualityReport.

        Args:
            table_fqn: Fully-qualified table name.
            failed_report: The QualityReport that did not pass quality gates.

        Returns:
            Dict of suggested parameter overrides (e.g. increased epoch count,
            alternate strategy).
        """
        suggestions: dict[str, Any] = {}
        cfg = get_settings()

        # KS failures → more epochs or switch to TVAE
        low_ks_cols = [col for col, p in failed_report.ks_test_results.items() if p < 0.05]
        if low_ks_cols:
            suggestions["ctgan_epochs"] = cfg.ctgan_epochs + 200
            suggestions["reason"] = f"KS test failed on columns: {low_ks_cols}"
            # If we already tried CTGAN and it failed, suggest TVAE
            runs_for_table = self.history.get(table_fqn, [])
            ctgan_failures = [
                r for r in runs_for_table
                if r["strategy"] == "ctgan" and not r.get("overall_pass")
            ]
            if len(ctgan_failures) >= 1:
                suggestions["switch_strategy"] = "tvae"

        # High correlation delta → Copula often captures correlations better
        if failed_report.correlation_delta_frobenius > 0.3:
            suggestions["switch_strategy"] = "copula"
            suggestions["correlation_reason"] = (
                f"Frobenius norm {failed_report.correlation_delta_frobenius:.3f} exceeds 0.3"
            )

        return suggestions
