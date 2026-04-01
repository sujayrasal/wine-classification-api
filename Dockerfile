# ── Base image ──────────────────────────────────────
FROM python:3.9-slim

# ── Set working directory inside container ───────────
WORKDIR /app

# ── Copy requirements first (Docker cache optimization)
COPY requirements.txt .

# ── Install dependencies ──────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy all project files into container ────────────
COPY . .

# ── Expose port 8000 ──────────────────────────────────
EXPOSE 8000

# ── Start FastAPI server ───────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
