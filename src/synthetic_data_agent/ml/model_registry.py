import json
from pathlib import Path
from typing import Optional, Dict, Any
from ..models.quality_report import QualityReport
from ..config import settings
import structlog

logger = structlog.get_logger()

class ModelRegistry:
    """Tracks trained model performance and strategy selection."""
    
    def __init__(self, storage_path: Path = settings.model_storage_path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_path = self.storage_path / "model_history.json"
        self._load_history()

    def _load_history(self):
        if self.history_path.exists():
            with open(self.history_path, "r") as f:
                self.history = json.load(f)
        else:
            self.history = {}

    def _save_history(self):
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def record_run(self, table_fqn: str, strategy: str, quality_report: QualityReport):
        """Record the outcome of a generation run."""
        if table_fqn not in self.history:
            self.history[table_fqn] = []
            
        self.history[table_fqn].append({
            "strategy": strategy,
            "overall_pass": quality_report.overall_pass,
            "ks_test_avg": sum(quality_report.ks_test_results.values()) / len(quality_report.ks_test_results) if quality_report.ks_test_results else 0,
            "timestamp": quality_report.generated_at.isoformat()
        })
        self._save_history()

    def best_strategy_for(self, table_fqn: str) -> Optional[str]:
        """Returns best historical strategy based on quality pass rate."""
        if table_fqn not in self.history:
            return None
            
        # Simplistic approach: pick the strategy with highest average KS score
        strategies = {}
        for run in self.history[table_fqn]:
            strat = run["strategy"]
            score = run["ks_test_avg"]
            if strat not in strategies:
                strategies[strat] = []
            strategies[strat].append(score)
            
        if not strategies:
            return None
            
        best_strat = max(strategies.keys(), key=lambda s: sum(strategies[s])/len(strategies[s]))
        return best_strat

    def get_tuning_suggestions(self, table_fqn: str, failed_report: QualityReport) -> Dict[str, Any]:
        """Provide suggestions to improve the model based on failure details."""
        suggestions = {}
        # If KS tests are failing, suggest more epochs
        low_ks_cols = [col for col, score in failed_report.ks_test_results.items() if score < 0.05]
        if low_ks_cols:
            suggestions["ctgan_epochs"] = settings.ctgan_epochs + 200
            suggestions["reason"] = f"Low KS scores on columns: {low_ks_cols}"
            
        return suggestions
