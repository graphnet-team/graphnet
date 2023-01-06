# Define variables
EXAMPLE_DATA_DIR=training_example_data
BUCKET=https://user-web.icecube.wisc.edu/~asogaard/example_data
EXAMPLE_FILE_NAME=prometheus-1250-events

# Download data
mkdir -p $EXAMPLE_DATA_DIR
wget -P $EXAMPLE_DATA_DIR "$BUCKET/$EXAMPLE_FILE_NAME.db"
wget -P $EXAMPLE_DATA_DIR "$BUCKET/$EXAMPLE_FILE_NAME.parquet"