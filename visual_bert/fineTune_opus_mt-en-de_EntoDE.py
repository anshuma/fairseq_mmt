import os
from torch.utils.data import Dataset, DataLoader
data_dir = '../small_dataset/data/multi30k-en-de'
class TranslationDataset(Dataset):
    def __init__(self, split):
        with open(f'{data_dir}/{split}.en', 'r') as src_f, open(f'{data_dir}/{split}.en', 'r') as tgt_f:
            self.source_sentences = src_f.read().strip().split('\n')
            self.target_sentences = tgt_f.read().strip().split('\n')
        assert len(self.source_sentences) == len(self.target_sentences), "Mismatch between source and target sentence count"

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        return self.source_sentences[idx], self.target_sentences[idx]


import heapq

class BestModels:
    def __init__(self, k=5, verbose=False):
        self.k = k
        self.verbose = verbose
        self.best_models = []

    def __call__(self, score, model, tokenizer, epoch):
        if len(self.best_models) < self.k:
            heapq.heappush(self.best_models, (score, epoch, model, tokenizer))
            self.save_checkpoint(score, model, tokenizer, epoch)
        else:
            if score > self.best_models[0][0]:  # Compare with the smallest score in the heap
                _, _, old_model, old_tokenizer = heapq.heappop(self.best_models)
                heapq.heappush(self.best_models, (score, epoch, model, tokenizer))
                self.save_checkpoint(score, model, tokenizer, epoch)

    def save_checkpoint(self, score, model, tokenizer, epoch):
        """Saves model when validation score improves."""
        path = f"best_model_epoch_{epoch}_score_{score:.4f}"
        if self.verbose:
            print(f"Validation score improved. Saving model to {path}...")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)


import torch
from torch.optim import AdamW
from transformers import get_scheduler, MarianMTModel, MarianTokenizer, GenerationConfig
from sacrebleu import corpus_bleu
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluate_model(model, tokenizer, dataloader, device):
    model.eval()
    references = []
    hypotheses = []
    with torch.no_grad():
        for source_sentences, target_sentences in tqdm(dataloader):
            inputs = tokenizer(source_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
            translated_ids = model.generate(**inputs)
            translated_sentences = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
            references.extend(target_sentences)
            hypotheses.extend(translated_sentences)
    bleu_score = corpus_bleu(hypotheses, [references])
    return bleu_score

def train_translation_model(model, tokenizer, train_dataloader, valid_dataloader, optimizer, scheduler, device, generation_config, num_epochs=3, patience=3):
    best_models = BestModels(k=5, verbose=True)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for source_sentences, target_sentences in train_dataloader:
            inputs = tokenizer(source_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
            labels = tokenizer(target_sentences, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        # Evaluate on validation data
        bleu_score = evaluate_model(model, tokenizer, valid_dataloader, device)
        print(f"Validation BLEU score: {bleu_score.score}")
        # Save the best models
        best_models(bleu_score.score, model, tokenizer, epoch)

# Load the datasets
train_dataset = TranslationDataset('train')
valid_dataset = TranslationDataset('valid')
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# Load the model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-de'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
model.to(device)

# Set up the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Train the model
train_translation_model(model, tokenizer, train_dataloader, valid_dataloader, optimizer, scheduler, device, generation_config=None, num_epochs=num_epochs)

# Save the final model
model.save_pretrained('fine-tuned-translation-model_opus_mt-en-de_EntoDE')
tokenizer.save_pretrained('fine-tuned-translation-model_opus_mt-en-de_EntoDE-tokenizer')
