from logging import info
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from fastprogress import progress_bar
from skimage.io import imread
from skimage.transform import resize as imresize
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg16


# def read_img_batch(img_paths):
#     imgs = [read_image(img_paths[i], resize=True) for i in range(len(img_paths))]
#     return np.stack(imgs, axis=0)


def process_batch(model, imgs, processed, already_tensor=False):
    if not already_tensor:
        img_tensor = torch.tensor(imgs, dtype=torch.float32, device='cuda')
    else:
        img_tensor = imgs
    img_tensor = rearrange(img_tensor, "b h w c -> b c h w")
    with torch.no_grad():
        preproc_img = model(img_tensor).cpu().numpy()
    processed.append(preproc_img)
    return processed


class RawImagesDataset(Dataset):
    def __init__(self, img_files):
        self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        return read_image(self.img_files[idx], True)


def load_pretrained_feature_extractor():
    # TODO: better models
    model = vgg16(pretrained=True)
    model.classifier = model.classifier[:-3]
    model.eval()
    model.to("cuda")
    return model


def preprocess_images(imgs_dir, n_jobs=4, batch_size=128):
    info("Start preprocessing images")
    model = load_pretrained_feature_extractor()
    files = sorted(Path(imgs_dir).iterdir())
    preprocessed = []
    imgs_dataset = RawImagesDataset(files)
    dataloader = DataLoader(imgs_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=n_jobs)
    for batch in progress_bar(dataloader):
        process_batch(model, batch.cuda(), preprocessed, True)
    return [x.stem for x in files], np.concatenate(preprocessed)


def read_image(img_path, resize=False):
    image = imread(img_path)

    if len(image.shape) != 3:
        image = np.stack([image, image, image], axis=2)

    if resize:
        return imresize(image, (224, 224), mode='reflect', anti_aliasing=True).astype(np.float32)
    return image