#!/bin/bash

# Check if a command file is provided
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <command_file>"
    exit 1
fi

# Get the command file from the argument
command_file="$1"

# Check if the file exists
if [[ ! -f $command_file ]]; then
    echo "Error: File '$command_file' not found."
    exit 1
fi

# Read and execute each line in the file
while IFS= read -r line || [[ -n $line ]]; do
    if [[ -n $line ]]; then # Ensure the line is not empty
        echo "Executing: $line"
        eval "$line"
        if [[ $? -ne 0 ]]; then # Check for errors
            echo "Error while executing: $line"
        fi
    fi
done < "$command_file"

echo "All commands executed successfully."