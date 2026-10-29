# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

# Reference host for training and checking the Muninn reference models.
#
#   podman build -t muninn-reference-models .
#   podman run --rm -v $PWD:/work -w /work muninn-reference-models pytest
FROM docker.io/library/python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# tmu (comparison benchmark only) has C code that newer GCCs reject by default
RUN CFLAGS="-Wno-error=incompatible-pointer-types" \
    pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/work
