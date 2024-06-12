import os
from torch.utils.data import Dataset
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as transforms

data_dir = '../small_dataset/data/multi30k-en-de'
image_dir = '../small_dataset/flickr30k/flickr30k-images'
image_idx_dir = '../small_dataset/flickr30k'


from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import sacrebleu
from tqdm import tqdm

# Load pre-trained model and tokenizer
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base',do_resize=False)
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
model.train()  # Set the model to training mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

config = resolve_data_config({}, model=model)
#transform = create_transform(**config)

from torchvision.transforms import Compose, Resize, ToTensor
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
    """
    Resize an image to `(size["height"], size["width"])`.

    Args:
        image (`np.ndarray`):
            Image to resize.
        size (`Dict[str, int]`):
            Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output image. If unset, the channel dimension format of the input
            image is used. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

    Returns:
        `np.ndarray`: The resized image.
    """
    image=to_numpy_array(image)
    return resize(
        image,
        size=(384,384),
        resample=resample,
        data_format=data_format,
        input_data_format=input_data_format,
        **kwargs,
    )

'''
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
])
'''
#transform = resize


class GermanImageCaptionDataset(Dataset):
    def __init__(self, split, processor,transform=None):
        with open(f'{image_idx_dir}/{split}.txt', 'r') as img_f, open(f'{data_dir}/{split}.de', 'r') as cap_f:
            #fileName = img_f.read().strip().split('\n')
            self.image_filenames = img_f.read().strip().split('\n')
            self.captions = cap_f.read().strip().split('\n')
        assert len(self.image_filenames) == len(self.captions), "Mismatch between images and captions"
        #self.transform = transform
        self.transform = transform
        self.processor = processor

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


# Load the datasets
train_dataset = GermanImageCaptionDataset('train', processor)
valid_dataset = GermanImageCaptionDataset('valid', processor)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        images, captions = batch
        inputs = processor(images=images, text=captions, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}/{num_epochs} finished with loss: {loss.item()}")

    # Validation loop
    model.eval()
    references = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            images, refs = batch
            inputs = processor(images=images, return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs, forced_bos_token_id=processor.tokenizer.lang_code_to_id["de"])
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

            references.append(refs[0])
            predictions.append(preds[0])

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"Epoch {epoch+1} Validation BLEU score: {bleu.score}")

# Save the fine-tuned model
model.save_pretrained('fine-tuned-blip')
processor.save_pretrained('fine-tuned-blip')


# Evaluation and BLEU score calculation for a test set
test_dataset = GermanImageCaptionDataset('test',processor)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Switch model to evaluation mode
model.eval()

references = []
predictions = []

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        images, refs = batch
        inputs = processor(images=images, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, forced_bos_token_id=processor.tokenizer.lang_code_to_id["de"])
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

        references.append(refs[0])
        predictions.append(preds[0])

# Calculate BLEU score
bleu = sacrebleu.corpus_bleu(predictions, [references])
print(f"BLEU score: {bleu.score}")
