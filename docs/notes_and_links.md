# links: 
- [Pytorch quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [a basic binary image classifier](https://github.com/github/codespaces-jupyter/blob/main/notebooks/image-classifier.ipynb)
- [Hugging face datasets docs](https://huggingface.co/docs/datasets/load_hub)
- [imagenette dataset](https://huggingface.co/datasets/frgfm/imagenette)
- [DiffusionDB dataset](https://huggingface.co/datasets/poloclub/diffusiondb)


# Workloads:
- Datamanagement:
  - Implementation of our dataloader which should provied the following functionality:
    - Load imagenette and diffusionDB dataset from hugging face 
    - combine them into one dataset and create labels: 
        - diffusionDB: all images should be labeld as 1 (for fake)
        - imagenet: all images should be labeld as 0 (for real)
    - shuffle the dataset and split the into train and test set (80/20)
        - make sure we have about equal amount of real and fake images in the train and test set
    - create a torch dataloader which can be used for training 
- Resnet50
  - Research and implement/copy model implementation of Resnet50 for binary image classification
  - Are there pretrained weights?
  - What loss function, optimizer and learning rate should we use?
- Train and monitor model:
  - Implement training notebook
  - Test if training works (maybe use arbitrary dataset before ours is ready)
    - Figure out what GPU we coud use, how to run training in colab
    - Early stopping and checkpoint saving
- Test and evaluate model:
  - implement test notebook, calculate evaluation metrics
  - Feasable options for experiments?
  - How does overleaf work? Start outlining the report

# Splitting the workload:
- Either per person or everyone works on all parts whatever is open right now