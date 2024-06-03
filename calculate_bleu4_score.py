import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def tokenize(lines):
    return [line.strip().split() for line in lines]

def read_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Paths to the reference and hypothesis files
reference_file = 'small_dataset/data/multi30k-en-de/train.de'  # Path to the reference translation file
hypothesis_file = 'small_dataset/data/multi30k-en-de/train_src.de'    # Path to the translated text file

# Read and tokenize the reference and hypothesis files
#references = tokenize(read_from_file(reference_file))
#hypotheses = tokenize(read_from_file(hypothesis_file))

# Calculate BLEU-4 score
#print('len(list_of_references)',len(references))
#print('len(hypotheses)',len(hypotheses))
#bleu_score = corpus_bleu(references, hypotheses)
#print(f"BLEU-4 score: {bleu_score:.4f}")

references = read_from_file(reference_file)
hypotheses = read_from_file(hypothesis_file)

print('len(list_of_references)',len(references))
print('len(hypotheses)',len(hypotheses))
smoothie = SmoothingFunction().method4
bleu_scores = [
    sentence_bleu([ref], trans.split(), smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25))
    for ref, trans in zip(references, hypotheses)
]

avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"BLEU-4 Score: {avg_bleu_score:.4f}")
