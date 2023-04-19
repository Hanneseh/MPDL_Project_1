from datasets import load_dataset_builder, get_dataset_split_names, load_dataset
import torchvision

# explore the dataset
ds_builder = load_dataset_builder("poloclub/diffusiondb")
ds_builder.info.description
ds_builder.info.features
get_dataset_split_names("poloclub/diffusiondb", "2m_random_1k")

# load smalles subset 
dataset = load_dataset("poloclub/diffusiondb", "2m_random_1k", split="train")

# split into test and train
train_val_test_splits = dataset.train_test_split(test_size=0.3, seed=42)



### torc

torchvision.datasets.ImageNet(root='', train=True, download=True)