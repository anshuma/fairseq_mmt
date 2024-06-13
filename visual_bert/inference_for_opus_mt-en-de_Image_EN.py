import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer, CLIPProcessor, CLIPModel
from tqdm import tqdm
import sacrebleu


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

data_dir = '../data/data/multi30k-en-de'
image_dir = '../flickr30k/flickr30k-images'
image_idx_dir = '../flickr30k'

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


class TranslationModelWithFeatures(nn.Module):
    def __init__(self, translation_model, clip_model):
        super(TranslationModelWithFeatures, self).__init__()
        self.translation_model = translation_model
        self.clip_model = clip_model
        self.W1 = nn.Parameter(torch.randn(1))
        self.W2 = nn.Parameter(torch.randn(1))
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
            text_features = self.extract_text_features(text_model, text_tokenizer, texts, device)

        return image_features, clip_text_features, text_features

    def forward(self, clip_processor, images, texts, text_model, text_tokenizer, device):
        image_features, clip_text_features, text_features = self.extract_features(clip_processor, images, texts,
                                                                                  text_model, text_tokenizer, device)
        combined_features = self.W1 * image_features + self.W2 * clip_text_features + self.W3 * text_features
        return combined_features


def generate_caption(model, clip_processor, image, text, text_model, text_tokenizer, device):
    combined_features = model(clip_processor, [image], [text], text_model, text_tokenizer, device)
    generated_ids = model.translation_model.generate(inputs_embeds=combined_features.unsqueeze(1))
    generated_text = text_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def inference(model, clip_processor, text_model, text_tokenizer, dataloader, device):
    model.eval()
    model.clip_model.eval()
    text_model.eval()

    references = []
    hypotheses = []

    with torch.no_grad():
        for images, text_captions in tqdm(dataloader):
            for i in range(len(images)):
                image = images[i].to(device)
                text_caption = text_captions[i]
                generated_caption = generate_caption(model, clip_processor, image, text_caption, text_model,
                                                     text_tokenizer, device)

                references.append([text_caption])
                hypotheses.append(generated_caption)

    # Calculate BLEU score using sacrebleu
    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"BLEU score: {bleu_score.score}")


# Example usage:

# Load the saved model checkpoint
checkpoint_path = 'path_to_save_model_checkpoint.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

translation_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
translation_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
#text_model = MarianMTModel.from_pretrained('path_to_save_huggingface_model').to(device)
#text_tokenizer = MarianTokenizer.from_pretrained('path_to_save_tokenizer')
text_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de').to(device)
text_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# Initialize the custom model
model = TranslationModelWithFeatures(translation_model, clip_model).to(device)

# Load state_dict
#model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Create a DataLoader for the test data
# Example dummy data for testing
#images = torch.randn(10, 3, 224, 224)  # 10 random images
#text_captions = ["This is a test caption."] * 10  # 10 identical English captions

# Create dataset and dataloaders
dataset = ImageTextDataset('test.2016')
test_dataloader = DataLoader(dataset, batch_size=2)

# Perform inference
inference(model, clip_processor, text_model, text_tokenizer, test_dataloader, device)
