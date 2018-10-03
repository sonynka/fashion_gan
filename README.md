# FashionGAN Search

This project is part of the evaluation of generative adversarial networks for improving image retrieval systems. It uses a fashion dataset to synthesize new images of fashion products based on user input, and to trigger a search of similar existing products.
The main application allows user to modify the shape and pattern of a dress, and then choose the best match from the retrieved products. The user can then further modify the chosen product.

![results](https://raw.githubusercontent.com/sonynka/fashion_gan/images/results.png)

## Usage

#### App
To use the project run the *FashionGAN_search.ipynb* notebook in the given conda environment (see requirements). The application started in the notebook prompts the user to control the image modifications and search by text input.

#### Processing
The notebooks in processing folder were used to download the feature vectors for image retrieval and clustering model images. All the data that they produce is already provided in the data folder. However, these notebooks can be run to further understand these processing steps.

#### Networks
The networks folder contains the three generators used in the final model
- **StarGAN** originally from: https://github.com/yunjey/StarGAN
- **CycleGAN** originally from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- **Pix2Pix** originally from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

The networks were trained on the fashion dataset, and the best models are provided in the data folder.

## Setup

#### Data
All data neccessary for running this project can be downloaded by running the following script:

```bash
cd data
./download_data.sh
```

The script will download the following folders:
- **images**: images of dresses and models wearing those dresses that the networks were trained on (cca 15.000 product images and 60.000 model images).
- **clustering**: models and data for clustering of available model images to create a paired dataset of 1 product image + 1 model image
- **features**: feature vectors for both product images and clustered model images for retrieval
- **models**: trained GAN models to modify attributes of images (The models were trained using several GANs repositories: **Pix2Pix** and **CycleGAN** on https://github.com/sonynka/pytorch-CycleGAN-and-pix2pix and **StarGAN** on https://github.com/sonynka/StarGAN.

**Note**: The original dataset was scraped from various fashion online stores and contains cca 90.000 images. For the purpose of this project, I only used category dresses. Code for scraping and the whole dataset can be found here: https://github.com/sonynka/fashion_scraper.

#### Requirements
To download Anaconda package manager, go to: <i>https://www.continuum.io/downloads</i>.
After installing the conda environment locally, proceed to setup this project environment.

Install all dependencies from conda_requirements.txt file.
```bash
conda create -n fashion_gan python=3.6
source activate fashion_gan
conda install --file conda_requirements.txt
pip install -r pip_requirements.txt
```

To start a jupyter notebook in the environment:
```bash
source activate fashion_gan
jupyter notebook
```


To deactivate this specific virtual environment:
```bash
source deactivate
```

If you need to completely remove this conda env, you can use the following command:

```bash
conda env remove --name fashion_gan
```
