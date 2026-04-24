# Synthetic Data Agent (SDA)

A production-grade synthetic data generation system built with **Google ADK** (Agent Development Kit). SDA learns statistical and structural patterns from complex datasets and generates high-fidelity synthetic data using state-of-the-art neural networks (CTGAN, TVAE, TimeGAN) while maintaining ironclad PII isolation.

---

## 🛠️ Local Setup

### 1. Prerequisites
- **Python 3.13+** (Optimized for latest features)
- **uv** (Recommended for blazing fast dependency management)
- **Redis** (Used for the FK Registry and Business Rule Knowledge Base)
- **Databricks Account** (Optional: local file uploads are supported natively)

### 2. Install Dependencies
Using `uv` for environment setup:
```bash
# Create venv and install project + dev dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
databricks_host="your-databricks-host"
databricks_token="your-token"
databricks_catalog="source_catalog"
databricks_schema="source_schema"
output_catalog="synthetic_output_catalog"
gemini_api_key="your-google-gemini-api-key"
redis_url="redis://localhost:6379"
```

---

## 🧪 Testing Locally

### 1. Run Unit Tests
We use `pytest` with `pytest-asyncio` for non-blocking I/O tests.
```bash
# Run all unit tests (PII, Registry, Generators)
pytest tests/unit/
```

### 2. Start the Agent Server
The server uses ADK's `get_fastapi_app` for native A2A support and agent discovery.
```bash
# Start the FastAPI server on port 8000
python -m synthetic_data_agent.server.main
```

### 3. Verify Endpoints
- **Health Check:** `curl http://localhost:8000/health`
- **Agent Discovery:** `curl http://localhost:8000/orchestrator` (Check if `root_agent` is active)
- **File Upload:**
```bash
curl -X POST -F "file=@sample_data.csv" http://localhost:8000/upload
```

---

## 🤖 Pipeline Workflow

1. **Upload / Connect:** Upload a CSV/Parquet file or connect to Databricks.
2. **Interactive Rule Setting:** Tell the Orchestrator about constraints:
   *   *"Remember that for the payments table, transaction_amount must be greater than 0."*
3. **Execution:** The Orchestrator triggers the specialists:
    *   **Profiler:** Learns distributions and flags PII.
    *   **EntityGraph:** Maps Foreign Keys.
    *   **Generator:** Trains CTGAN/TVAE on non-PII data + applies memory rules.
    *   **PIIHandler:** Generates provably non-real PII (Isolated pipeline).
    *   **Validator:** Runs KS tests to ensure distribution fidelity.

---

## 🏗️ Architecture Best Practices
- **PII Isolation:** PII data is never fed into Neural Networks. It is handled by a separate agent using format-preserving synthetic generators.
- **Topological Generation:** Parent tables (e.g., Users) are generated before child tables (e.g., Transactions) to maintain referential integrity via the Redis ID Registry.
- **Adaptive Tuning:** The `ModelRegistry` tracks KS scores and automatically tunes epochs or switches strategies (e.g., CTGAN -> TVAE) if quality gates fail.
