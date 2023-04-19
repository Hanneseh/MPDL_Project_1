# Notes:
- hugging face account is required for downloading imagenet dataset


# links: 
- [Pytorch quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [a basic binary image classifier](https://github.com/github/codespaces-jupyter/blob/main/notebooks/image-classifier.ipynb)
- [Hugging face datasets docs](https://huggingface.co/docs/datasets/load_hub)
- [ImageNet dataset](https://huggingface.co/datasets/imagenet-1k)
- [DiffusionDB dataset](https://huggingface.co/datasets/poloclub/diffusiondb)


# Workloads:

- Datamanagement:
  - Implementation of our dataloader which should provied the following functionality:
    - Load imagenet and diffusionDB dataset from hugging face (maybe with streaming flag if too big to store in colab later on)
    - combine them into one dataset and create labels: 
        - diffusionDB: all images should be labeld as 1 (for fake)
        - imagenet: all images should be labeld as 0 (for real)
    - shuffle the dataset and split the into train and test set (80/20)
        - make sure we have about equal amount of real and fake images in the train and test set
    - create a torch dataloader which can be used for training 
- Resnet50
  - Research and implement/copy model implementation of Resnet50 for binary image classification
  - Are there pretrained weights?
  - Test if training works (maybe use arbitrary dataset before ours is ready)
    - Figure out what GPU we coud use, how to run training in colab
- Test and evaluate model:
  - implement test notebook, calculate evaluation metrics
  - How does overleaf work? Start outlining the report

# Splitting the workload:
- Either per person or everyone works on all parts whatever is open right now