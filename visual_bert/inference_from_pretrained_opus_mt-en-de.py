import torch
import nltk
from transformers import MarianMTModel, MarianTokenizer
from sacrebleu import corpus_bleu
#from nltk.translate.meteor_score import meteor_score
#from vizseq.scorers.meteor import METEORScorer
# Load the pretrained MarianMT model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-de'
#model_name = 'model_opus_mt-en-de_EntoDE_epoch_3_score_99.8978'
#model_name = 'model_opus_mt-en-de_EntoDE_epoch_1_score_99.6013'
#model_name = 'model_opus_mt-en-de_EntoDE_epoch_7_score_42.5981'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

#nltk.download('wordnet')
#nltk.download('omw-1.4')
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data_dir = '../data/multi30k-en-de'
#data_dir = '../final_data'
def read_sentences(filename):
    with open(f'{data_dir}/{filename}', 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file]
    return sentences


def translate_sentences(model, tokenizer, sentences, device):
    translations = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(device)
        translated_ids = model.generate(**inputs)
        translated_sentence = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        translations.append(translated_sentence)
    return translations


def calculate_bleu(testname):
    # Read English input sentences and German reference sentences
    english_sentences = read_sentences(testname+'.en')
    german_references = read_sentences(testname+'.de')
    # Translate the English sentences to German
    translated_sentences = translate_sentences(model, tokenizer, english_sentences, device)

    # Prepare reference format for sacrebleu
    references = [[ref] for ref in german_references]

    # Calculate BLEU score
    #print('len(references)',len(references))
    #print('len(translated_sentences)',len(translated_sentences))
    #bleu = corpus_bleu(translated_sentences, [german_references]) : for scrableu 1.5.1
    bleu = corpus_bleu(translated_sentences, references)
    #meteor_scores = [meteor_score([ref.split()], hyp.split()) for ref, hyp in zip(german_references, translated_sentences)]
    #avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    print('TEST NAME:',testname)
    print(f"BLEU score: {bleu.score}")
    #print(f"METEOR score: {avg_meteor_score*100}")



calculate_bleu('test.2016')
calculate_bleu('test.2017')
#calculate_bleu('test.2018')
'''
tokenized_references = [ref.split() for ref in german_references]
tokenized_hypotheses = [hyp.split() for hyp in translated_sentences]
meteor_score = METEORScorer(sent_level=False, corpus_level=True).score(
        tokenized_hypotheses, [tokenized_references]
    )
print('meteor score',meteor_score)
'''
