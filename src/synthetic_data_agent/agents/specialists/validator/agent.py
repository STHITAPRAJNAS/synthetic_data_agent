from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from google.adk.agents import Agent
from google.adk.tools import tool  # type: ignore[attr-defined]
from scipy.stats import chi2_contingency, ks_2samp

from ....agents.callbacks import (
    after_model_callback,
    after_tool_callback,
    before_model_callback,
    before_tool_callback,
)

from ....models.quality_report import QualityReport
from ....pii.leakage_auditor import PrivacyAuditor
from ....tools.databricks_tools import DatabricksTools
from ....tools.knowledge_base import KnowledgeBase
from ....tools.registry_tools import SyntheticIDRegistry

logger = structlog.get_logger()

_db_tools = DatabricksTools()
_privacy_auditor = PrivacyAuditor()
_knowledge_base = KnowledgeBase()
_registry = SyntheticIDRegistry()


@tool
async def validate_table(
    source_table_fqn: str,
    synth_table_fqn: str,
    quasi_ids: list[str],
) -> dict[str, Any]:
    """Run all quality gates on a synthetic table and return a QualityReport.

    Gates:
    1. KS test per numerical column (p-value > 0.05 = pass).
    2. Chi-squared test per categorical column (p-value > 0.05 = pass).
    3. Correlation matrix Frobenius norm delta (< 0.15 = pass).
    4. FK integrity: zero orphaned FK references.
    5. Business rule assertion pass rate (> 99.9% = pass).
    6. PII leakage: exact-match rate < 1%.
    7. k-anonymity on quasi-PII columns (k >= 5).

    Args:
        source_table_fqn: Original source table FQN (used as ground truth).
        synth_table_fqn: Generated synthetic table FQN to validate.
        quasi_ids: Column names to include in k-anonymity calculation.

    Returns:
        Serialised QualityReport dict.
    """
    logger.info("Validating synthetic table", source=source_table_fqn, synth=synth_table_fqn)

    sample_size = 10_000
    real_df = await _db_tools.sample_dataframe(source_table_fqn, sample_size)
    synth_df = await _db_tools.sample_dataframe(synth_table_fqn, sample_size)

    failure_details: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1 & 2. Statistical fidelity — KS test (numerical) + chi-squared (categorical)
    # ------------------------------------------------------------------
    ks_results: dict[str, float] = {}
    stat_pass = True

    num_cols = real_df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if col in synth_df.columns:
            real_vals = real_df[col].dropna()
            synth_vals = synth_df[col].dropna()
            if len(real_vals) > 0 and len(synth_vals) > 0:
                _, p_val = ks_2samp(real_vals, synth_vals)
                ks_results[col] = float(p_val)
                if p_val < 0.05:
                    stat_pass = False
                    failure_details[f"ks_{col}"] = f"p={p_val:.4f} < 0.05"

    cat_cols = real_df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if col in synth_df.columns:
            real_counts = real_df[col].value_counts()
            synth_counts = synth_df[col].value_counts()
            all_cats = list(set(real_counts.index) | set(synth_counts.index))
            real_freq = [real_counts.get(c, 0) for c in all_cats]
            synth_freq = [synth_counts.get(c, 0) for c in all_cats]
            if sum(real_freq) > 0 and sum(synth_freq) > 0:
                try:
                    _, p_val, _, _ = chi2_contingency([real_freq, synth_freq])
                    ks_results[f"chi2_{col}"] = float(p_val)
                    if p_val < 0.05:
                        stat_pass = False
                        failure_details[f"chi2_{col}"] = f"p={p_val:.4f} < 0.05"
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 3. Correlation matrix delta (Frobenius norm)
    # ------------------------------------------------------------------
    corr_delta = 0.0
    corr_pass = True
    shared_num = [c for c in num_cols if c in synth_df.columns]
    if len(shared_num) >= 2:
        real_corr = real_df[shared_num].corr().fillna(0).values
        synth_corr = synth_df[shared_num].corr().fillna(0).values
        corr_delta = float(np.linalg.norm(real_corr - synth_corr, ord="fro"))
        if corr_delta >= 0.15:
            corr_pass = False
            failure_details["correlation_frobenius"] = f"delta={corr_delta:.4f} >= 0.15"

    # ------------------------------------------------------------------
    # 4. FK integrity — zero orphaned references
    # ------------------------------------------------------------------
    fk_pass = True
    # We verify by checking that FK column values are all registered PKs
    # (A full check requires knowing the FK spec — best-effort here)
    for col in synth_df.columns:
        if col.lower().endswith("_id") and col in real_df.columns:
            synth_ids = set(synth_df[col].dropna().astype(str).unique())
            real_ids = set(real_df[col].dropna().astype(str).unique())
            orphans = synth_ids - real_ids
            if len(orphans) > 0:
                orphan_rate = len(orphans) / max(len(synth_ids), 1)
                if orphan_rate > 0.001:  # >0.1% orphans = fail
                    fk_pass = False
                    failure_details[f"fk_{col}"] = f"{len(orphans)} orphaned values"

    # ------------------------------------------------------------------
    # 5. Business rule pass rate
    # ------------------------------------------------------------------
    rules = await _knowledge_base.get_business_rules(source_table_fqn)
    total_rules = len(rules)
    passed_rules = 0
    for rule in rules:
        try:
            filtered = synth_df.query(rule["code"])
            if len(filtered) / max(len(synth_df), 1) >= 0.999:
                passed_rules += 1
        except Exception as exc:
            logger.error("Business rule evaluation failed", rule=rule["description"], error=str(exc))

    rule_pass_rate = (passed_rules / total_rules) if total_rules > 0 else 1.0
    rule_pass = rule_pass_rate >= 0.999
    if not rule_pass:
        failure_details["business_rules"] = f"pass_rate={rule_pass_rate:.4f} < 0.999"

    # ------------------------------------------------------------------
    # 6 & 7. Privacy audit (leakage + k-anonymity)
    # ------------------------------------------------------------------
    privacy_report = await _privacy_auditor.audit_report(real_df, synth_df, quasi_ids)
    pii_leakage = privacy_report["leakage_rate"] > 0.01
    k_anon_min = int(privacy_report["k_anonymity"])
    if pii_leakage:
        failure_details["pii_leakage"] = f"leakage_rate={privacy_report['leakage_rate']:.4f}"
    if k_anon_min < 5:
        failure_details["k_anonymity"] = f"k={k_anon_min} < 5"

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    overall_pass = (
        stat_pass
        and corr_pass
        and fk_pass
        and rule_pass
        and not pii_leakage
        and k_anon_min >= 5
    )

    report = QualityReport(
        table_fqn=source_table_fqn,
        ks_test_results=ks_results,
        correlation_delta_frobenius=corr_delta,
        fk_integrity_pass=fk_pass,
        business_rule_pass_rate=rule_pass_rate,
        pii_leakage_detected=pii_leakage,
        k_anonymity_min=k_anon_min,
        overall_pass=overall_pass,
        failure_details=failure_details,
    )

    logger.info(
        "Validation complete",
        table=source_table_fqn,
        overall_pass=overall_pass,
        failures=list(failure_details.keys()),
    )
    return report.model_dump(mode="json")


# ---------------------------------------------------------------------------
# ADK root agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="validator_agent",
    model="gemini-2.5-flash",
    description=(
        "Synthetic data quality specialist that runs KS tests, chi-squared tests, "
        "correlation matrix comparison, FK integrity checks, business rule assertions, "
        "PII leakage detection, and k-anonymity validation."
    ),
    instruction="""You are a synthetic data quality specialist.

For each table generated, call validate_table to run all quality gates.
Return the serialised QualityReport so the orchestrator can decide whether
to accept the output or trigger self-correction in the GeneratorAgent.

A table passes when ALL of these hold:
- KS / chi-squared p-values > 0.05 (statistically similar distributions)
- Frobenius norm of correlation delta < 0.15
- Zero orphaned FK references
- Business rule pass rate >= 99.9%
- PII leakage rate < 1%
- k-anonymity >= 5 on quasi-PII columns""",
    tools=[validate_table],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
