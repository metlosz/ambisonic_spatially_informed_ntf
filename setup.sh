#!/bin/bash
#coding=utf-8

set -euo pipefail

# Get the scipt directory (this will fail for symbolic link as last path component)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

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
