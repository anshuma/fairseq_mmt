import nltk
from nltk.translate.bleu_score import corpus_bleu

def tokenize(lines):
    return [line.strip().split() for line in lines]

def read_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Paths to the reference and hypothesis files
reference_file = 'reference.txt'  # Path to the reference translation file
hypothesis_file = 'output.txt'    # Path to the translated text file

# Read and tokenize the reference and hypothesis files
references = [tokenize(read_from_file(reference_file))]
hypotheses = tokenize(read_from_file(hypothesis_file))

# Calculate BLEU-4 score
bleu_score = corpus_bleu(references, hypotheses)
print(f"BLEU-4 score: {bleu_score:.4f}")
