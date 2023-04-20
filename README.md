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

## Tasks
- [x] Train baseline binary image classifier on starting Dataset: ResNet50 trained on DiffusionDB "2m_first_10k" ("train") subset and Imagenette "full", "train"
  - [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb)
  - [Imagenette](https://huggingface.co/datasets/frgfm/imagenette)
- [ ] Perform further experiments 
- [ ] Write report

## Helpful Links
- [Pytorch quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [basic classifier](https://github.com/github/codespaces-jupyter/blob/main/notebooks/image-classifier.ipynb)
- [Hugging face datasets docs](https://huggingface.co/docs/datasets/load_hub)