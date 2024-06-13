import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer, MBartForConditionalGeneration, MBartTokenizer, CLIPProcessor, \
    CLIPModel, GenerationConfig
from sacrebleu import corpus_bleu
from tqdm import tqdm
import os
from PIL import Image
import heapq
from typing import Dict, List, Optional, Union
import numpy as np
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.image_transforms import resize
def transform(
        image: np.ndarray,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
) -> np.ndarray:

    image=to_numpy_array(image)
    return resize(
        image,
        size=(384,384),
        resample=resample,
        data_format=data_format,
        input_data_format=input_data_format,
        **kwargs,
    )

data_dir = '../small_dataset/data/multi30k-en-de'
image_dir = '../small_dataset/flickr30k/flickr30k-images'
image_idx_dir = '../small_dataset/flickr30k'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ImageTextDataset(Dataset):
    def __init__(self, split):
        with open(f'{image_idx_dir}/{split}.txt', 'r') as img_f, open(f'{data_dir}/{split}.en', 'r') as src_cap_f,open(f'{data_dir}/{split}.de', 'r') as cap_f:
            # fileName = img_f.read().strip().split('\n')
            self.image_filenames = img_f.read().strip().split('\n')
            self.text_captions = src_cap_f.read().strip().split('\n')
            self.captions = cap_f.read().strip().split('\n')
        assert len(self.image_filenames) == len(self.captions), "Mismatch between images and captions"

    def __len__(self):
        return len(self.image_filenames)

    #def __getitem__(self, idx):
    #    return self.images[idx], self.captions[idx], self.text_captions[idx]

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(image_dir, img_filename)
        caption = self.captions[idx]
        text_captions = self.text_captions[idx]
        image = Image.open(img_path).convert('RGB')
        #if self.transform:
        #    image = self.transform(image)
        #else:
        image = transform(image)

        return image, caption, text_captions

class BestModels:
    def __init__(self, k=10, verbose=False):
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


# Load models and tokenizers
translation_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
translation_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

translation_model.to(device)
clip_model.to(device)
# Load text model and tokenizer (MarianMTModel or MBART)
text_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de').to(device)
text_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
class TranslationModelWithFeatures(nn.Module):
    def __init__(self):
        super(TranslationModelWithFeatures, self).__init__()
        self.translation_model = translation_model
        self.clip_model = clip_model
        self.translation_tokenizer=translation_tokenizer
        self.W1 = nn.Parameter(torch.randn(1))
        #self.W2 = nn.Parameter(torch.randn(1))
        self.W3 = nn.Parameter(torch.randn(1))

    def extract_text_features(self, text_model, tokenizer, texts, device):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = text_model.model.encoder(**inputs)
            text_features = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling
        return text_features

    def extract_features(self, clip_processor, images, texts, text_model, text_tokenizer, device):
        # Process images
        image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
        # Process texts for CLIP
        text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**image_inputs)
            clip_text_features = self.clip_model.get_text_features(**text_inputs)
            '''
            text_features = text_model.model.encoder(
                input_ids=text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(
                    device)).last_hidden_state.mean(dim=1)
            '''
            text_features = self.extract_text_features(text_model, text_tokenizer, texts, device)

        return image_features, clip_text_features, text_features

    def forward(self, clip_processor, images, texts, text_model, text_tokenizer, device):
        image_features, clip_text_features, text_features = self.extract_features(clip_processor, images, texts,
                                                                                  text_model, text_tokenizer, device)
        #combined_features = self.W1 * image_features + self.W2 * clip_text_features + self.W3 * text_features
        #combined_features = self.W1 * image_features + self.W3 * text_features
        combined_features = self.W1 * image_features.unsqueeze(1).repeat(1, text_features.size(1), 1) + \
                            self.W3 * text_features.unsqueeze(1).repeat(1, text_features.size(1), 1)
        return combined_features


# Create dataset and dataloaders
train_dataset = ImageTextDataset('train')
valid_dataset = ImageTextDataset('valid')
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)


# Uncomment below lines if using MBart
# text_model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50').to(device)
# text_tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-50')

# Define generation config
generation_config = GenerationConfig(
    max_length=512,
    num_beams=4,
    bad_words_ids=[[58100]],
    forced_eos_token_id=0,
    decoder_start_token_id=translation_tokenizer.pad_token_id  # Use pad_token_id as decoder_start_token_id
)

# Move models to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mbart_model.to(device)  # Uncomment if using MBartForConditionalGeneration

# Initialize custom model
model = TranslationModelWithFeatures().to(device)


def generate_caption(model, clip_processor, image, text, text_model, text_tokenizer, generation_config):
    image = image.to(device)
    combined_features = model(clip_processor, [image], [text], text_model, text_tokenizer, device)

    #generated_ids = model.translation_model.generate(inputs_embeds=combined_features.unsqueeze(1))
    generated_ids = model.translation_model.generate(inputs_embeds=combined_features)
    generated_text = model.translation_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def evaluate_model(model, clip_processor, text_model, text_tokenizer, dataloader, generation_config):
    model.eval()
    clip_model.eval()
    text_model.eval()

    references = []
    hypotheses = []

    with torch.no_grad():
        for images, captions, text_captions in tqdm(dataloader):
            for i in range(len(images)):
                image = images[i]
                caption = captions[i]
                text_caption = text_captions[i]
                generated_caption = generate_caption(model, clip_processor, image, text_caption, text_model,
                                                     text_tokenizer, generation_config)

                references.append([caption])
                hypotheses.append(generated_caption)

    bleu_score = corpus_bleu(hypotheses, references)
    print(f"BLEU score: {bleu_score.score}")
    return bleu_score


def train_translation_model(model, clip_processor, text_model, text_tokenizer, train_dataloader, valid_dataloader,
                            optimizer, scheduler, device, generation_config, num_epochs=30):
    best_models = BestModels(k=10, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            images, captions, text_captions = batch
            images = images.to(device)
            combined_features = model(clip_processor, images, text_captions, text_model, text_tokenizer, device)

            inputs = model.translation_tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(
                device)
            labels = inputs.input_ids

            optimizer.zero_grad()
            outputs = model.translation_model(inputs_embeds=combined_features, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Evaluate on validation data
        bleu_score = evaluate_model(model, clip_processor, text_model, text_tokenizer, valid_dataloader,
                                    generation_config)

        # Save the best models
        best_models(bleu_score.score, model.translation_model, model.translation_tokenizer, epoch)


# Define optimizer and scheduler
'''
optimizer = torch.optim.Adam([
    {'params': model.translation_model.parameters()},
    {'params': [model.W1, model.W2, model.W3]}
], lr=5e-5)
'''
optimizer = torch.optim.Adam([
    {'params': model.translation_model.parameters()},
    {'params': [model.W1, model.W3]}
], lr=5e-5)
num_training_steps = 30 * len(train_dataloader)  # Example number of epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_training_steps // 3, gamma=0.1)

# Train the model
train_translation_model(model, clip_processor, text_model, text_tokenizer, train_dataloader, valid_dataloader,
                        optimizer, scheduler, device, generation_config, num_epochs=30)
