#!/bin/bash
set -e
jupyter lab --port $JUPYTER_PORT --ip 0.0.0.0 --allow-root --no-browser --NotebookApp.token=$TOKEN