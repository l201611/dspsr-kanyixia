#!/usr/bin/env bash

# Function to recursively compare files in subdirectories
compare_files() {
    local baseline="$1"
    local newpipelinetests="$2"

    # Loop through each file in the baseline directory
    find "$baseline" -type f -print0 | while IFS= read -r -d '' baseline_file; do
        # Extract the relative path of the file
        relative_path="${baseline_file#$baseline/}"
        new_file="$newpipelinetests/$relative_path"

        # Check if the corresponding file exists in newpipelinetests
        if [ -f "$new_file" ]; then
            # Print the common file and their details
            echo "Common file found: $relative_path"
            echo "Baseline file:"
            ls -l "$baseline_file"
            echo "New pipeline tests file:"
            ls -l "$new_file"
            echo "psrdiff -X Result:"
            psrdiff -X $baseline_file $new_file
            echo ""
        fi
    done
}

# Accept baseline and newpipelinetests paths as input parameters
baseline="$1"
newpipelinetests="$2"

# Create an output file
output_file="output.txt"
echo "Output:" > "$output_file"

# Compare files recursively
compare_files "$baseline" "$newpipelinetests" >> "$output_file"

echo "Output saved to $output_file"
