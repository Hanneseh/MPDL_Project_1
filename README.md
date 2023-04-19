# MPDL_Poject_1
Project repo for "Masterpraktikum Deep Learning and Natural Language Processing" at the Ruhr University Bochum in the summer term 2023.

## Project Description

Goal: given some image, determine whether it’s real or generated using Stable Diffusion
Timeline:
1. Go over pytorch basics (1 week)
2. Implement baseline binary image classifier (1 week)
   - starting dataset: DiffusionDB subset for fake, Imagenette for real 
   - starting model: Resnet50
3. Run experiments + write report (1 week) 
   - remember to use train/valid/test split
4. Try improve (1 week):
   - more data (OpenImages?)
   - other models (Vision transformer?)
   - perturbations (blur, JPEG, crop, zoom, ...)

## Development Environment
1. Develop code locally, run training on Google Colab
   - This tutorial [tutorial](https://felixbmuller.medium.com/connect-a-private-github-repository-with-google-colab-via-a-deploy-key-cca8ad13007) does not work for me.
2. Using GitHub Codespaces (I am currently waiting on GPU access)
   1. [How to use](https://docs.github.com/en/codespaces/developing-in-codespaces/getting-started-with-github-codespaces-for-machine-learning)

## Tasks
- [ ] Implement code to use the [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb) dataset
  - [ ] How to connect to the dataset on huggingface? What size of subset should we use?
  - [ ] How to combine it with the Imagenette dataset? (this is available in [torchvision.datasets](https://pytorch.org/vision/stable/datasets.html))
  - [ ] Implement splitting into train/valid/test
  - [ ] Write functions to load data with pytorch torchvision.datasets (look at)
- [ ] Implement [Resnet50](https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/) (use pretrained weights, use existing implementation)
- [ ] Train and log results
- [ ] Evaluate model on test set
- [ ] Write report

## Helpful Links
- [Pytorch quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [basic classifier](https://github.com/github/codespaces-jupyter/blob/main/notebooks/image-classifier.ipynb)
- [Hugging face datasets docs](https://huggingface.co/docs/datasets/load_hub)