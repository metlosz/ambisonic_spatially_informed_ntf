#!/bin/bash
#coding=utf-8

# Set error behavior
set -euo pipefail

# Get script directory
SCRIPT_DIR=$(realpath $(dirname $0))

# Initialize conda
CONDA_DIR=$(conda info --base)
source $CONDA_DIR/etc/profile.d/conda.sh

# Update conda
conda update conda

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate asintf

# Install the package in editable mode
pip install -e $SCRIPT_DIR
