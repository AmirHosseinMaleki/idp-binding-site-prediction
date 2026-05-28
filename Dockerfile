# ─────────────────────────────────────────────────────────────────────────────
# IDP Binding Site Predictor - CAID submission image
#
# Build:
#   docker build -t idp-binding-caid .
#
# Run:
#   docker run \
#     -v /path/to/sequences.fasta:/input/sequences.fasta \
#     -v /path/to/embeddings:/embeddings \
#     -v /path/to/output:/output \
#     idp-binding-caid \
#       --fasta /input/sequences.fasta \
#       --embeddings_dir /embeddings \
#       --output_dir /output
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.8-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-caid.txt .
RUN pip install --no-cache-dir -r requirements-caid.txt

COPY predict_caid.py .

# model weights - 3 files, live in data/ by default
RUN mkdir -p /app/data
COPY data/protein_hybrid_idpval_model.pt  /app/data/
COPY data/dna_rna_hybrid_idpval_model.pt  /app/data/
COPY data/ion_hybrid_idpval_model.pt      /app/data/

# CAID will mount embeddings and output at runtime
VOLUME ["/input", "/embeddings", "/output"]

ENTRYPOINT ["python", "predict_caid.py"]
CMD ["--help"]