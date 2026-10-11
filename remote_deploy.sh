#!/usr/bin/env bash
# Run on the server as root by setup_remote_server after the sync, from $SERVICE_DIR.
# Env: SERVICE SERVICE_DIR SERVICE_ROOT BASE SERVICE_USER SERVICE_GROUP; helper: as_user <cmd...>
#
# /static is on PATH, so this name is the command for every account.
set -euo pipefail

chmod 755 "$SERVICE_DIR/bin/predictor"
ln -sfn "$SERVICE_DIR/bin/predictor" /static/predict

# The install tree is not writable. Every account stores the cache here.
mkdir -p /var/cache/predictors
chmod 1777 /var/cache/predictors
