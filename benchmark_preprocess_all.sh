#!/bin/bash

# Define the folder containing the Python scripts
PREPROCESS_FOLDER="./benchmark/maintenance/preprocess"

# Check if the folder exists
if [ ! -d "$PREPROCESS_FOLDER" ]; then
  echo "Error: Folder $PREPROCESS_FOLDER does not exist." > benchmark_preprocess_all.log
  exit 1
fi

# Iterate through all .py files in the folder, excluding the specified ones
echo "" > benchmark_preprocess_all.log
for file in "$PREPROCESS_FOLDER"/*.py; do
  filename=$(basename "$file")
  if [[ "$filename" != "__init__.py" && "$filename" != "dataset_utils.py" ]]; then
    echo "Executing $file..." >> benchmark_preprocess_all.log
    python "$file" >> benchmark_preprocess_all.log
    if [ $? -ne 0 ]; then
      echo "Error: Execution of $file failed." >> benchmark_preprocess_all.log
      exit 1
    fi
  fi
done

echo "All eligible Python files executed successfully." >> benchmark_preprocess_all.log
