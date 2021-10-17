#!/usr/bin/env python3
import sys

from assignment import load_model, compute_perplexity

if __name__ == '__main__':
    model_filename = sys.argv[1]
    test_filename = sys.argv[2]
    LM = load_model(model_filename)
    perplexity = compute_perplexity(LM, test_filename, 3)
    print("Perplexity Report:")
    print("------------------")
    print("Model File Name: " + model_filename)
    print("Test File Name: " + test_filename)
    print("Perplexity: " + str(perplexity))
