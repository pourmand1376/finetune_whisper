#!/bin/bash
set -e
jupyter lab --port 8888 --ip 0.0.0.0 --allow-root --no-browser --NotebookApp.token=docker