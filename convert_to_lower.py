# Specify input and output file paths
input_file_path = "dataset/data/task1/raw/train.de"
output_file_path = "finalRawData/train.ed"

# Open the input file for reading and the output file for writing
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        # Convert each line to lowercase and write to the output file
        outfile.write(line.lower())

