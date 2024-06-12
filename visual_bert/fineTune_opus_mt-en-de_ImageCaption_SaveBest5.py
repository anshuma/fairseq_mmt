import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer, CLIPProcessor, CLIPModel, GenerationConfig
from sacrebleu import corpus_bleu
from tqdm import tqdm
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
class GermanImageCaptionDataset(Dataset):
    def __init__(self, split, transform=None):
        with open(f'{image_idx_dir}/{split}.txt', 'r') as img_f, open(f'{data_dir}/{split}.de', 'r') as cap_f:
            # fileName = img_f.read().strip().split('\n')
            self.image_filenames = img_f.read().strip().split('\n')
            self.captions = cap_f.read().strip().split('\n')
        assert len(self.image_filenames) == len(self.captions), "Mismatch between images and captions"
        '''
        self.transform = transform if transform else Compose([
            Resize((224, 224)),
            ToTensor(),
        ])
        '''
        self.transform=transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(image_dir, img_filename)
        caption = self.captions[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = transform(image)

        return image, caption
# Define BestModels class
class BestModels:
    def __init__(self, k=7, verbose=False):
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


# Load the fine-tuned models and tokenizer
translation_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
translation_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# Define generation config
generation_config = GenerationConfig(
    max_length=512,
    num_beams=4,
    bad_words_ids=[[58100]],
    forced_eos_token_id=0,
    decoder_start_token_id=translation_tokenizer.pad_token_id  # Use pad_token_id as decoder_start_token_id
)

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translation_model.to(device)
clip_model.to(device)


def extract_features(clip_model, clip_processor, images):
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features


def generate_caption(translation_model, translation_tokenizer, clip_model, clip_processor, image, generation_config):
    image = image.to(device)
    features = extract_features(clip_model, clip_processor, image.unsqueeze(0))

    generated_ids = translation_model.generate(inputs_embeds=features.unsqueeze(1))
    generated_text = translation_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def evaluate_model(translation_model, translation_tokenizer, clip_model, clip_processor, dataloader, generation_config):
    translation_model.eval()
    clip_model.eval()

    references = []
    hypotheses = []

    with torch.no_grad():
        for images, captions in tqdm(dataloader):
            for i in range(len(images)):
                image = images[i]
                caption = captions[i]
                generated_caption = generate_caption(translation_model, translation_tokenizer, clip_model,
                                                     clip_processor, image, generation_config)

                references.append([caption])
                hypotheses.append(generated_caption)

    bleu_score = corpus_bleu(hypotheses, references)
    print(f"BLEU score: {bleu_score.score}")
    return bleu_score


def train_translation_model(translation_model, translation_tokenizer, clip_model, clip_processor, train_dataloader,
                            valid_dataloader, optimizer, scheduler, device, generation_config, num_epochs=30):
    best_models = BestModels(k=7, verbose=True)

    for epoch in range(num_epochs):
        translation_model.train()
        total_loss = 0
        for batch in train_dataloader:
            images, captions = batch
            images = images.to(device)
            features = extract_features(clip_model, clip_processor, images)

            inputs = translation_tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
            labels = inputs.input_ids
            inputs_embeds = features.unsqueeze(1).repeat(1, labels.size(1), 1)

            optimizer.zero_grad()
            outputs = translation_model(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Evaluate on validation data
        bleu_score = evaluate_model(translation_model, translation_tokenizer, clip_model, clip_processor,
                                    valid_dataloader, generation_config)

        # Save the best models
        best_models(bleu_score.score, translation_model, translation_tokenizer, epoch)


# Load the datasets
base_dir = '/path/to/data'
train_dataset = GermanImageCaptionDataset('train')
valid_dataset = GermanImageCaptionDataset('valid')
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

import torch
from torch.optim import AdamW
from transformers import get_scheduler
# Set up the optimizer and scheduler
optimizer = AdamW(translation_model.parameters(), lr=5e-5)
num_epochs = 30
num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Train the model
train_translation_model(translation_model, translation_tokenizer, clip_model, clip_processor, train_dataloader,
                        valid_dataloader, optimizer, scheduler, device, generation_config, num_epochs=num_epochs)

# Save the final model
translation_model.save_pretrained('final-fine-tuned-translation-model')
translation_tokenizer.save_pretrained('final-fine-tuned-translation-tokenizer')
