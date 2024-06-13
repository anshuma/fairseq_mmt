import os
import nltk
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer, CLIPProcessor, CLIPModel
from sacrebleu import corpus_bleu
from tqdm import tqdm
#from nltk.translate.meteor_score import meteor_score
#from vizseq.scorers.meteor import METEORScorer

#nltk.download('wordnet')
#nltk.download('omw-1.4')
data_dir = '../data/data/multi30k-en-de'
#image_dir = '../flickr30k/flickr30k-images'
image_idx_dir = '../flickr30k'

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
    def __init__(self, split, image_dir, transform=None):
        with open(f'{image_idx_dir}/{split}.txt', 'r') as img_f, open(f'{data_dir}/{split}.de', 'r') as cap_f:
            # fileName = img_f.read().strip().split('\n')
            self.image_filenames = img_f.read().strip().split('\n')
            self.captions = cap_f.read().strip().split('\n')
        assert len(self.image_filenames) == len(self.captions), "Mismatch between images and captions"
        self.image_dir = image_dir
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
        img_path = os.path.join(self.image_dir, img_filename)
        caption = self.captions[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = transform(image)

        return image, caption


# Load the fine-tuned models and tokenizer
translation_model = MarianMTModel.from_pretrained('opus_mt_en_de_checkpoints/best_model_with_bleuscore_64.75')
translation_tokenizer = MarianTokenizer.from_pretrained('opus_mt_en_de_checkpoints/best_model_with_bleuscore_64.75')
#translation_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
#translation_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device',device)
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


def evaluate_model(translation_model, translation_tokenizer, clip_model, clip_processor, dataloader,testname):
    translation_model.eval()
    clip_model.eval()

    references = []
    hypotheses = []
    met_references = []

    with torch.no_grad():
        for images, captions in tqdm(dataloader):
            for i in range(len(images)):
                image = images[i]
                caption = captions[i]
                generated_caption = generate_caption(translation_model, translation_tokenizer, clip_model,
                                                     clip_processor, image)

                references.append([caption])
                met_references.append(caption)
                hypotheses.append(generated_caption)

    #bleu_score = corpus_bleu(hypotheses, references)
    #print(f"BLEU score: {bleu_score.score}")
    #bleu = corpus_bleu(hypotheses,[met_references]) #older scrableu 1.5.1
    bleu = corpus_bleu(hypotheses,references) #scrableu 2.4.2
    #meteor_scores = [meteor_score([ref.split()], hyp.split()) for ref, hyp in zip(references, hypotheses)]
    #avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    print('TEST NAME:',testname)
    print(f"BLEU score: {bleu.score}")
    '''
    tokenized_references = [ref.split() for ref in met_references]
    tokenized_hypotheses = [hyp.split() for hyp in hypotheses]

    meteor_scores = [meteor_score([ref], hyp) for ref, hyp in zip(tokenized_references, tokenized_hypotheses)]
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    avg_meteor_score_percentage = avg_meteor_score * 100
    print(f"METEOR score: {avg_meteor_score_percentage:.2f}%")
    #print(f"METEOR score: {avg_meteor_score*100}")
    '''
    return bleu


# Load the test dataset
#base_dir = '/path/to/data'
def calculate_bleu(testname,image_dir):
    test_dataset = GermanImageCaptionDataset(testname,image_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate the model
    evaluate_model(translation_model, translation_tokenizer, clip_model, clip_processor, test_dataloader, testname)

#test2017-images
calculate_bleu('test.2016','../flickr30k/flickr30k-images')
calculate_bleu('test.2017','../flickr30k/test2017-images')
#calculate_bleu('test.2018','../flickr30k/test2018-images')
