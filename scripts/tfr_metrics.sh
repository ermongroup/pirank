SCRIPTS_DIR=$(dirname $(readlink -fm "$0"))
TFR_FILE=$(python -c 'import tensorflow_ranking as tfr; print(tfr.metrics.__file__)' 2> /dev/null)
cd $(dirname "$TFR_FILE")
patch < "$SCRIPTS_DIR"/tfr_metrics.patch