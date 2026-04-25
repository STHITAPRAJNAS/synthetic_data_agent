"""Model quality registry — tracks generation run history and drives adaptation.

Responsibilities
-----------------
1. Record the outcome (QualityReport) of every completed generation attempt.
2. Persist history to disk so it survives across agent sessions.
3. Recommend the historically best-performing strategy for a given table.
4. Produce tuning suggestions when a run fails quality gates.
5. Record which artifact key holds the currently best model for a table so the
   generator agent can load it from the ADK artifact store without retraining.

Design: append-only JSON history + in-memory index.

The file is written asynchronously (asyncio.to_thread) so it never blocks the
event loop, and it is never truncated — only appended — to avoid data loss on
concurrent writes.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import structlog

from ..config import get_settings
from ..models.quality_report import QualityReport

logger = structlog.get_logger()

_STRATEGIES = ("ctgan", "tvae", "copula", "timegan")


class ModelRegistry:
    """Adaptive quality history for all synthesis models.

    All public methods are safe to call from async code.  ``__init__`` loads
    existing history synchronously — it is intended to be called once at
    module import time, before any event loop is running.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or get_settings().model_storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_path = self.storage_path / "model_history.json"
        self._history: dict[str, list[dict[str, Any]]] = self._load_sync()
        # table_fqn → artifact_key of the best stored model
        self._best_artifact: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Sync bootstrap
    # ------------------------------------------------------------------

    def _load_sync(self) -> dict[str, list[dict[str, Any]]]:
        if self.history_path.exists():
            try:
                with open(self.history_path) as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("model_history_load_failed", error=str(exc))
        return {}

    # ------------------------------------------------------------------
    # Async persistence
    # ------------------------------------------------------------------

    async def _save(self) -> None:
        """Persist history to disk without blocking the event loop."""
        data = json.dumps(self._history, indent=2, default=str)
        try:
            await asyncio.to_thread(self.history_path.write_text, data, encoding="utf-8")
        except Exception as exc:
            logger.error("model_history_save_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record_run(
        self,
        table_fqn: str,
        strategy: str,
        quality_report: QualityReport,
        artifact_key: str | None = None,
    ) -> None:
        """Append the outcome of one generation attempt.

        Args:
            table_fqn: Fully-qualified table name.
            strategy: Strategy used ('ctgan', 'tvae', 'copula', 'timegan').
            quality_report: Result from ValidatorAgent.
            artifact_key: ADK artifact key for the trained model, if saved.
        """
        if table_fqn not in self._history:
            self._history[table_fqn] = []

        ks_vals = list(quality_report.ks_test_results.values())
        ks_avg = sum(ks_vals) / len(ks_vals) if ks_vals else 0.0

        entry: dict[str, Any] = {
            "strategy": strategy,
            "overall_pass": quality_report.overall_pass,
            "ks_test_avg": round(ks_avg, 4),
            "correlation_delta": round(quality_report.correlation_delta_frobenius, 4),
            "pii_leakage": quality_report.pii_leakage_detected,
            "k_anonymity_min": quality_report.k_anonymity_min,
            "business_rule_pass_rate": round(quality_report.business_rule_pass_rate, 4),
            "timestamp": quality_report.generated_at.isoformat(),
        }
        if artifact_key:
            entry["artifact_key"] = artifact_key

        self._history[table_fqn].append(entry)

        # Track best artifact for this table
        if quality_report.overall_pass and artifact_key:
            self._best_artifact[table_fqn] = {
                "artifact_key": artifact_key,
                "strategy": strategy,
                "ks_test_avg": str(ks_avg),
            }

        await self._save()
        logger.info(
            "model_run_recorded",
            table=table_fqn,
            strategy=strategy,
            passed=quality_report.overall_pass,
            ks_avg=round(ks_avg, 3),
        )

    def best_strategy_for(self, table_fqn: str) -> str | None:
        """Return the historically best strategy for *table_fqn*, or None on first run.

        Prefer strategies that passed all quality gates; break ties by highest
        average KS p-value (higher = better statistical similarity).
        """
        runs = self._history.get(table_fqn)
        if not runs:
            return None

        passing = [r for r in runs if r.get("overall_pass")]
        candidates = passing if passing else runs

        strategy_scores: dict[str, list[float]] = {}
        for run in candidates:
            s = run["strategy"]
            strategy_scores.setdefault(s, []).append(run.get("ks_test_avg", 0.0))

        if not strategy_scores:
            return None

        best = max(
            strategy_scores,
            key=lambda s: sum(strategy_scores[s]) / len(strategy_scores[s]),
        )
        logger.info("best_historical_strategy", table=table_fqn, strategy=best)
        return best

    def best_artifact_for(self, table_fqn: str) -> dict[str, str] | None:
        """Return artifact metadata for the best stored model, or None.

        Returns a dict with keys ``artifact_key`` and ``strategy``.
        """
        return self._best_artifact.get(table_fqn)

    def get_tuning_suggestions(
        self,
        table_fqn: str,
        failed_report: QualityReport,
    ) -> dict[str, Any]:
        """Derive parameter adjustments from a failed QualityReport.

        Returns a dict of suggested overrides.  Keys consumed by the generator:
          - ``switch_strategy``: try a different ML strategy
          - ``epochs_boost``: add this many extra epochs to the current strategy
          - ``reason``: human-readable explanation for agent instruction
        """
        suggestions: dict[str, Any] = {}
        cfg = get_settings()

        runs = self._history.get(table_fqn, [])
        strategies_tried = {r["strategy"] for r in runs if not r.get("overall_pass")}

        # KS failure → distribution mismatch → more capacity or different model
        low_ks = [c for c, p in failed_report.ks_test_results.items() if p < 0.05]
        if low_ks:
            suggestions["reason"] = f"KS test failed on {len(low_ks)} columns: {low_ks[:3]}"
            if "ctgan" in strategies_tried and "tvae" not in strategies_tried:
                suggestions["switch_strategy"] = "tvae"
            elif "tvae" in strategies_tried and "ctgan" not in strategies_tried:
                suggestions["switch_strategy"] = "ctgan"
            else:
                suggestions["epochs_boost"] = 150
                suggestions["switch_strategy"] = (
                    "tvae" if "ctgan" not in strategies_tried else "ctgan"
                )

        # High correlation delta → copula preserves pairwise correlations better
        if failed_report.correlation_delta_frobenius > 0.25 and "copula" not in strategies_tried:
            suggestions["switch_strategy"] = "copula"
            suggestions["reason"] = (
                f"Frobenius norm {failed_report.correlation_delta_frobenius:.3f} > 0.25; "
                "Copula may better preserve correlations."
            )

        # PII leakage → reduce fidelity, use copula (less likely to memorise)
        if failed_report.pii_leakage_detected and "copula" not in strategies_tried:
            suggestions["switch_strategy"] = "copula"
            suggestions["privacy_reason"] = "PII leakage detected; switching to Copula reduces memorisation."

        return suggestions

    def run_count(self, table_fqn: str) -> int:
        """Total number of recorded runs for a table."""
        return len(self._history.get(table_fqn, []))

    def summary(self) -> dict[str, Any]:
        """Return a compact summary of the whole registry (for /metrics)."""
        return {
            "tables_tracked": len(self._history),
            "total_runs": sum(len(v) for v in self._history.values()),
            "tables_with_passing_run": sum(
                1 for runs in self._history.values()
                if any(r.get("overall_pass") for r in runs)
            ),
        }
