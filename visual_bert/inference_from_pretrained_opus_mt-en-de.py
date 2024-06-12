import torch
from transformers import MarianMTModel, MarianTokenizer
from sacrebleu import corpus_bleu

# Load the pretrained MarianMT model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-de'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data_dir = '../small_dataset/data/multi30k-en-de'
def read_sentences(filename):
    with open(f'{data_dir}/{filename}', 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file]
    return sentences

# Read English input sentences and German reference sentences
english_sentences = read_sentences('train.en')
german_references = read_sentences('train.de')

def translate_sentences(model, tokenizer, sentences, device):
    translations = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(device)
        translated_ids = model.generate(**inputs)
        translated_sentence = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        translations.append(translated_sentence)
    return translations

# Translate the English sentences to German
translated_sentences = translate_sentences(model, tokenizer, english_sentences, device)

# Prepare reference format for sacrebleu
references = [[ref] for ref in german_references]

# Calculate BLEU score
bleu = corpus_bleu(translated_sentences, references)
print(f"BLEU score: {bleu.score}")
