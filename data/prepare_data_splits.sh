#!/bin/bash

# Script to prepare data splits by creating my_data_splits folder,
# copying input file to train.json and test.json, then running split_tests

# Check if data path argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No data path provided"
    echo "Usage: $0 <data_path>"
    echo "Example: $0 data_splits/codeforces.json"
    exit 1
fi

# Get the data path from command line argument
DATA_PATH="$1"

# Check if the input file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: File '$DATA_PATH' does not exist"
    exit 1
fi

# Create EASY_DATA_PATH in the same directory as DATA_PATH with filename lcb_easy.json
EASY_DATA_PATH="$(dirname "$DATA_PATH")/lcb_easy.json"
echo "Derived EASY_DATA_PATH: $EASY_DATA_PATH"

# Create my_data_easy folder if it doesn't exist
MY_DATA_EASY_DIR="${MY_DATA_EASY_DIR:-my_data_easy}"
echo "Creating directory: $MY_DATA_EASY_DIR"
mkdir -p "$MY_DATA_EASY_DIR"

# Define file paths for train.json and test.json
TRAIN_FILE_PATH="$MY_DATA_EASY_DIR/train.json"
TEST_FILE_PATH="$MY_DATA_EASY_DIR/test.json"

# Copy the original file to both train.json and test.json
echo "Copying $EASY_DATA_PATH to $TRAIN_FILE_PATH"
cp "$EASY_DATA_PATH" "$TRAIN_FILE_PATH"

echo "Copying $EASY_DATA_PATH to $TEST_FILE_PATH"
cp "$EASY_DATA_PATH" "$TEST_FILE_PATH"

# Check if the copies were successful
if [ ! -f "$TRAIN_FILE_PATH" ]; then
    echo "Error: Failed to create $TRAIN_FILE_PATH"
    exit 1
fi

if [ ! -f "$TEST_FILE_PATH" ]; then
    echo "Error: Failed to create $TEST_FILE_PATH"
    exit 1
fi

echo "Successfully created copies:"
echo "  - $TRAIN_FILE_PATH"
echo "  - $TEST_FILE_PATH"

# Run split_tests.main on the train.json file
echo "Running split_tests on $TRAIN_FILE_PATH"
python data/split_tests.py --data_path "$TRAIN_FILE_PATH"

# Run preprocessing on the my_data_easy folder
echo "Running preprocessing on my_data_easy folder"
python data/preprocess.py --data_source "$MY_DATA_EASY_DIR"

# Create my_data_splits folder if it doesn't exist
MY_DATA_SPLITS_DIR="${MY_DATA_SPLITS_DIR:-my_data_splits}"
echo "Creating directory: $MY_DATA_SPLITS_DIR"
mkdir -p "$MY_DATA_SPLITS_DIR"

# Define file paths for train.json and test.json
TRAIN_FILE_PATH="$MY_DATA_SPLITS_DIR/train.json"
TEST_FILE_PATH="$MY_DATA_SPLITS_DIR/test.json"

# Copy the original file to both train.json and test.json
echo "Copying $DATA_PATH to $TRAIN_FILE_PATH"
cp "$DATA_PATH" "$TRAIN_FILE_PATH"

echo "Copying $DATA_PATH to $TEST_FILE_PATH"
cp "$DATA_PATH" "$TEST_FILE_PATH"

# Check if the copies were successful
if [ ! -f "$TRAIN_FILE_PATH" ]; then
    echo "Error: Failed to create $TRAIN_FILE_PATH"
    exit 1
fi

if [ ! -f "$TEST_FILE_PATH" ]; then
    echo "Error: Failed to create $TEST_FILE_PATH"
    exit 1
fi

echo "Successfully created copies:"
echo "  - $TRAIN_FILE_PATH"
echo "  - $TEST_FILE_PATH"

# Run split_tests.main on the train.json file
echo "Running split_tests on $TRAIN_FILE_PATH"
python data/split_tests.py --data_path "$TRAIN_FILE_PATH"

# Run preprocessing on the my_data_splits folder
echo "Running preprocessing on my_data_splits folder"
python data/preprocess.py --data_source "$MY_DATA_SPLITS_DIR"

# Create my_data_singles folder structure with individual questions
MY_DATA_SINGLES_DIR="${MY_DATA_SINGLES_DIR:-my_data_singles}"
echo "Creating directory: $MY_DATA_SINGLES_DIR"
mkdir -p "$MY_DATA_SINGLES_DIR"

# Get the number of lines (questions) in train.json to determine how many folders to create
TRAIN_LINES=$(wc -l < "$TRAIN_FILE_PATH")
TEST_LINES=$(wc -l < "$TEST_FILE_PATH")

echo "Found $TRAIN_LINES questions in train.json and $TEST_LINES questions in test.json"

# Check if train and test have the same number of lines
if [ "$TRAIN_LINES" -ne "$TEST_LINES" ]; then
    echo "Error: train.json and test.json must have the same number of lines"
    echo "  train.json has $TRAIN_LINES lines"
    echo "  test.json has $TEST_LINES lines"
    echo "Please ensure both files contain the same number of questions"
    exit 1
fi

# Create individual question folders and extract single lines
for i in $(seq 0 $((TRAIN_LINES - 1))); do
    # Format folder name with zero padding (q_00, q_01, etc.)
    FOLDER_NAME=$(printf "q_%02d" $i)
    QUESTION_DIR="$MY_DATA_SINGLES_DIR/$FOLDER_NAME"
    
    echo "Creating question folder: $QUESTION_DIR"
    mkdir -p "$QUESTION_DIR"
    
    # Extract the (i+1)th line from train.json and save it 4 times
    TRAIN_SINGLE_PATH="$QUESTION_DIR/train.json"
    sed -n "$((i + 1))p" "$TRAIN_FILE_PATH" > "$TRAIN_SINGLE_PATH"
    sed -n "$((i + 1))p" "$TRAIN_FILE_PATH" >> "$TRAIN_SINGLE_PATH"
    sed -n "$((i + 1))p" "$TRAIN_FILE_PATH" >> "$TRAIN_SINGLE_PATH"
    sed -n "$((i + 1))p" "$TRAIN_FILE_PATH" >> "$TRAIN_SINGLE_PATH"
    
    # Extract the (i+1)th line from test.json and save it
    TEST_SINGLE_PATH="$QUESTION_DIR/test.json"
    sed -n "$((i + 1))p" "$TEST_FILE_PATH" > "$TEST_SINGLE_PATH"
    
    echo "  Created: $TRAIN_SINGLE_PATH"
    echo "  Created: $TEST_SINGLE_PATH"
    
    # Run preprocessing on this question folder
    echo "  Processing question folder: $QUESTION_DIR"
    python data/preprocess.py --data_source "$QUESTION_DIR"
done

echo "Successfully created $MY_DATA_SINGLES_DIR with individual question folders"

echo "Data preparation completed successfully!"