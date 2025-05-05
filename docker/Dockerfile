FROM mambaorg/micromamba:2.0

LABEL version="0.1.0"
LABEL software="CleanSweep"
LABEL software.version="CleanSweep v0.1"
LABEL description="Image containing CleanSweep v0.1 and dependencies."
LABEL maintainer="Marco Teixeira"
LABEL maintainer.email="mcarvalh@broadinstitute.org"
LABEL maintainer.institution="Broad Institute of MIT and Harvard"

# Copy requirements file 
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt

# Copy CleanSweep
RUN mkdir /tmp/cleansweep
COPY --chown=$MAMBA_USER:$MAMBA_USER cleansweep /tmp/cleansweep/cleansweep 
COPY --chown=$MAMBA_USER:$MAMBA_USER setup.py /tmp/cleansweep 
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml /tmp/cleansweep 
COPY --chown=$MAMBA_USER:$MAMBA_USER LICENSE /tmp/cleansweep 
COPY --chown=$MAMBA_USER:$MAMBA_USER README.md /tmp/cleansweep 

 
# Setup environment
RUN micromamba install -y -n base -f /tmp/requirements.txt && \
    micromamba clean --all --yes;

ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENV PATH="/opt/conda/bin:$PATH"

# Install Flit
RUN python3 -m pip install flit;

# Install setuptools
RUN micromamba install -y setuptools

# Install CleanSweep
RUN cd /tmp/cleansweep && flit install -s

RUN cleansweep --help