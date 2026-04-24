# Use an official Python runtime with CUDA support for ML models
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Install Python 3.13 and uv
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md .env ./

# Install project dependencies
RUN uv venv && source .venv/bin/activate && uv pip install .

# Copy source code
COPY src/ ./src/

# Expose FastAPI port
EXPOSE 8000

# Start the ADK Agent server
CMD ["source", ".venv/bin/activate", "&&", "python", "-m", "synthetic_data_agent.server.main"]
