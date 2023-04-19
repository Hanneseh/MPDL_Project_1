'''
- Implementation of our dataloader which should provied the following functionality:
    - Load imagenette and diffusionDB dataset from hugging face 
    - combine them into one dataset and create labels: 
        - diffusionDB: all images should be labeld as 1 (for fake)
        - imagenet: all images should be labeld as 0 (for real)
    - shuffle the dataset and split the into train and test set (80/20)
        - make sure we have about equal amount of real and fake images in the train and test set
    - create a torch dataloader which can be used for training 
'''


import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader

imagenette_dataset = datasets.load_dataset("frgfm/imagenette", "full_size", streaming=True, split="train")
diffusion_db_dataset = datasets.load_dataset("poloclub/diffusiondb", "2m_all", streaming=True, split="train")

# use with torch.utils.data.DataLoader
imagenette_dataset = imagenette_dataset.with_format("torch")
dataloader = DataLoader(imagenette_dataset, batch_size=4)

EPOCHS = 2
print("Training...")
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1} of {EPOCHS}", leave=True, ncols=80)):
        inputs, labels = data
        break
    break