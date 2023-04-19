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