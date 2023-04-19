# https://huggingface.co/docs/datasets/use_with_pytorch
# imagenet_1k_dataset = datasets.load_dataset("imagenet-1k", split="train")

import datasets
import numpy as np

imagenette_dataset = datasets.load_dataset("frgfm/imagenette", "full_size", split="train")
diffusion_db_dataset = datasets.load_dataset("poloclub/diffusiondb", "2m_random_10k", split="train")

# now add the "label" feature to the dataset
# add the "label" feature to the dataset
diffusion_labels = np.zeros(10000, dtype=np.int32)
diffusion_labels += 1
diffusion_db_dataset = diffusion_db_dataset.add_column("label", diffusion_labels)

diffuision_columns_to_drop = ['prompt', 'seed', 'step', 'cfg', 'sampler', 'user_name', 'timestamp', 'image_nsfw', 'prompt_nsfw', 'width', 'height']

# from diffusion_db_dataset drop all features except for "image" and "label" and 'width', 'height',
diffusion_db_dataset = diffusion_db_dataset.remove_columns(diffuision_columns_to_drop)

# for imagenette_dataset drop "label"
imagenette_dataset = imagenette_dataset.remove_columns(['label'])
imagenette_label = np.zeros(9469, dtype=np.int32)
imagenette_dataset = imagenette_dataset.add_column("label", imagenette_label)

# now concatenate the datasets
dataset = datasets.concatenate_datasets([imagenette_dataset, diffusion_db_dataset])

# now shuffle the dataset
dataset = dataset.shuffle()

# get dataset size
dataset_size = len(dataset)

# split into train and test (80/20)
train_dataset = dataset.select(range(int(dataset_size * 0.8)))
test_dataset = dataset.select(range(int(dataset_size * 0.8), dataset_size))