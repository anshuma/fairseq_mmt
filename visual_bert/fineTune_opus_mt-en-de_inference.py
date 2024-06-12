import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer, CLIPProcessor, CLIPModel
from sacrebleu import corpus_bleu
from tqdm import tqdm

data_dir = '../small_dataset/data/multi30k-en-de'
image_dir = '../small_dataset/flickr30k/flickr30k-images'
image_idx_dir = '../small_dataset/flickr30k'

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
    image = to_numpy_array(image)
    return resize(
        image,
        size=(384, 384),
        resample=resample,
        data_format=data_format,
        input_data_format=input_data_format,
        **kwargs,
    )


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
        self.transform = transform

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


# Load the fine-tuned models and tokenizer
translation_model = MarianMTModel.from_pretrained('fine-tuned-translation-model')
translation_tokenizer = MarianTokenizer.from_pretrained('fine-tuned-translation-tokenizer')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translation_model.to(device)
clip_model.to(device)


def extract_features(clip_model, clip_processor, images):
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features


def generate_caption(translation_model, translation_tokenizer, clip_model, clip_processor, image):
    image = image.to(device)
    features = extract_features(clip_model, clip_processor, image.unsqueeze(0))

    generated_ids = translation_model.generate(inputs_embeds=features.unsqueeze(1))
    generated_text = translation_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def evaluate_model(translation_model, translation_tokenizer, clip_model, clip_processor, dataloader):
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
                                                     clip_processor, image)

                references.append([caption])
                hypotheses.append(generated_caption)

    bleu_score = corpus_bleu(hypotheses, references)
    print(f"BLEU score: {bleu_score.score}")
    return bleu_score


# Load the test dataset
#base_dir = '/path/to/data'
test_dataset = GermanImageCaptionDataset('test.2016')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate the model
evaluate_model(translation_model, translation_tokenizer, clip_model, clip_processor, test_dataloader)
