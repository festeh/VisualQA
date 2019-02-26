import json
import logging
from logging import info
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from skimage.io import imread
from skimage.transform import resize as imresize
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm

from src.utils.helpers import create_parent_dir_if_not_exists, init_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_image(img_path, resize=False):
    image = Image.open(img_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')
    #     image = np.stack([image, image, image], axis=2)
    #
    if resize:
        return imresize(image, (224, 224), mode='reflect', anti_aliasing=True).astype(np.float32)
    return image


def process_batch(model, imgs, processed):
    # if not already_tensor:
    #     img_tensor = torch.tensor(imgs, dtype=torch.float32, device='cuda')
    # else:
    #     img_tensor = imgs
    # img_tensor = rearrange(img_tensor, "b h w c -> b c h w")
    with torch.no_grad():
        preproc_img = model(imgs).cpu().numpy()
    processed.append(preproc_img)
    return processed


class RawImagesDataset(Dataset):
    def __init__(self, img_files):
        self.img_files = img_files
        self.pipeline = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = read_image(self.img_files[idx], False)
        processed_image = self.pipeline(img)
        return idx, processed_image


def load_pretrained_feature_extractor():
    # TODO: better models
    model = vgg16(pretrained=True)
    model.classifier = model.classifier[:-3]
    model.eval()
    model.to("cuda")
    return model


def preprocess_images(
        train_images, val_images,
        train_images_result_file, val_images_result_file,
        train_filenames_result_file, val_filenames_result_file,
        n_jobs=4,
        batch_size=128):
    info("Start preprocessing raw images")
    model = load_pretrained_feature_extractor()

    def preprocess_dataset(images_path, filenames_saving_path, images_saving_path):
        files = sorted(Path(images_path).iterdir())
        preprocessed = []
        imgs_dataset = RawImagesDataset(files)
        dataloader = DataLoader(imgs_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=n_jobs)
        all_idxs = []
        for idxs, batch in tqdm(dataloader):
            all_idxs.extend(idxs)
            process_batch(model, batch.cuda(), preprocessed)
        assert sorted(all_idxs) == all_idxs
        filenames = [x.stem for x in files]
        images = np.concatenate(preprocessed)

        filenames_saving_path = Path(filenames_saving_path)
        create_parent_dir_if_not_exists(filenames_saving_path)
        with filenames_saving_path.open("w"):
            json.dump(filenames, filenames_saving_path.open("w"))

        images_saving_path = Path(images_saving_path)
        create_parent_dir_if_not_exists(images_saving_path)
        with h5py.File(images_saving_path, "w") as f:
            f.create_dataset("images", data=images, dtype='float32')
        return filenames, images

    preprocess_dataset(train_images, filenames_saving_path=train_filenames_result_file,
                       images_saving_path=train_images_result_file)
    preprocess_dataset(val_images, filenames_saving_path=val_filenames_result_file,
                       images_saving_path=val_images_result_file)
    info("Processed raw images!")


if __name__ == "__main__":
    preprocess_images(**init_config("data", preprocess_images))
