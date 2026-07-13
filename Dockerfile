FROM python:3.12-slim

WORKDIR /app

# unstructured's markdown parsing needs libmagic; keep the image lean otherwise.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# sentence-transformers pulls in torch, which by default fetches GPU/CUDA
# wheels on Linux -- several GB we don't need for CPU-only embedding of
# short text chunks. Install the CPU-only build first so it's already
# satisfied when requirements.txt is processed.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the index at image build time so container startup is fast.
# (Re-run manually after editing docs/ if you rebuild the image.)
RUN python -m scripts.build_index

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
