FROM mambaorg/micromamba

WORKDIR /app

# Datasets are mounted at runtime under /datasets/<task-name>
ENV BIOAGENT_DATA_ROOT=/datasets

# Pre-create dataset mount points for clarity (bind mounts will override)
RUN mkdir -p \
    /datasets/alzheimer-mouse \
    /datasets/comparative-genomics \
    /datasets/cystic-fibrosis \
    /datasets/deseq \
    /datasets/evolution \
    /datasets/giab \
    /datasets/metagenomics \
    /datasets/single-cell \
    /datasets/transcript-quant \
    /datasets/viral-metagenomics

# Declare volumes to document intent (host paths are provided at runtime)
VOLUME [ "/datasets/alzheimer-mouse", \
         "/datasets/comparative-genomics", \
         "/datasets/cystic-fibrosis", \
         "/datasets/deseq", \
         "/datasets/evolution", \
         "/datasets/giab", \
         "/datasets/metagenomics", \
         "/datasets/single-cell", \
         "/datasets/transcript-quant", \
         "/datasets/viral-metagenomics" ]