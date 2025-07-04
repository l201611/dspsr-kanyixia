ARG DSPSR_BASE_IMAGE=""
FROM $DSPSR_BASE_IMAGE as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PSRHOME=/home/pst
ENV CUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS} -diag-suppress 815,997 -Wno-deprecated-gpu-targets"
ARG UID=1000
ARG GID=1000
ARG UNAME=pst

# PREPARE ENVIRONMENT
USER root
RUN rm -rf $PSRHOME/src/dspsr $PSRHOME/build/dspsr \
    && mkdir -p $PSRHOME/src/dspsr $PSRHOME/build/dspsr \
    && chown ${UID}:${GID} $PSRHOME

COPY . $PSRHOME/src/dspsr

# Compile DSPSR
WORKDIR $PSRHOME/src/dspsr
RUN ./bootstrap
WORKDIR $PSRHOME/build/dspsr
RUN echo 'dada sigproc dummy fits vdif ska1 cpsr2 kat uwb' > backends.list
RUN $PSRHOME/src/dspsr/configure \
    --with-cuda-include-dir=/usr/local/cuda/include \
    --with-cuda-lib-dir=/usr/local/cuda/lib64 \
    --prefix=/usr/local/dspsr --enable-shared \
    && make -j$(nproc) \
    && make install \
    && chown -R $UID:$GID $PSRHOME
WORKDIR /home/${UNAME}

# Install pipeline test framework
WORKDIR $PSRHOME/src/dspsr/test/Pipeline
RUN python3 -m pip install poetry && \
  poetry install && \
  poetry run python -m pipeline -h

CMD ["/bin/bash"]
