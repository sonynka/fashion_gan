# FashionGAN Search

This project was created as part of my master thesis to evaluate the possibilities of generative adversarial networks to improve image retrieval.

## Usage

### App
To use the project run the FashionGAN_search notebook in the given conda environment (see requirements). The application started in the notebook can be used by entering text input.

### Processing
The notebooks in processing folder are used to download the feature vectors of the image search and clustering model images. All the data that they produce is already provided in the data folder.

## Setup

### Data
All data neccessary for running this project can be downloaded by running the following script:

```bash
cd data && download_data.sh
```

The script will download the following folders:
- **images**: images of dresses and models wearing those dresses that the networks were trained on (cca 15.000 product images and 60.000 model images).
- **clustering**: models and data for clustering of available model images to create a paired dataset of 1 product image + 1 model image
- **features**: feature vectors for both product images and clustered model images for retrieval
- **models**: trained GAN models to modify attributes of images (The models were trained using several GANs repositories: **Pix2Pix** and **CycleGAN** on https://github.com/sonynka/pytorch-CycleGAN-and-pix2pix and **StarGAN** on https://github.com/sonynka/StarGAN.

**Note**: The original dataset was scraped from various fashion online stores and contains cca 90.000 images. For the purpose of this project, I only used category dresses. Code for scraping and the whole dataset can be found here: https://github.com/sonynka/fashion_scraper.

### Requirements
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



