#!/usr/bin/env python3

from assignment import preprocess_line

# Models to Build mapped to the Files that we want to use to train then
{"our_model_eng": "training.en"}

# clean a source file
source = open("training.en", "r")
cleaned_lines = []

for line in source:
    cleaned_lines.append(preprocess_line(line))

source.close()

cleaned_file = open("cleaned_trianing.en", "w")
cleaned_file.writelines(cleaned_lines)
cleaned_file.close()
