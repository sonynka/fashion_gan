# FashionGAN Search

This project was created as part of my master thesis to evaluate the possibilities of generative adversarial networks to improve image retrieval.

# Prerequisites
## Data
All data neccessary for running this project can be download by running the following script:

```bash
cd data && download_data.sh
```

The script will download the following folders:
- **images**: images of fashion products and model wearing those products that the models were trained on (cca 15.000 product images and 60.000 model images). The original dataset for the training containing more than 90.000 images was scraped from online fashion websites (https://github.com/sonynka/fashion_scraper)
- **clustering**: models and data for clustering of available model images to create a paired dataset of 1 product image + 1 model image
- **features**: feature vectors for both product images and clustered model images for retrieval
- **models**: trained GAN models to modify attributes of images (The models were trained using several GANs repositories: **Pix2Pix** and **CycleGAN** on https://github.com/sonynka/pytorch-CycleGAN-and-pix2pix and **StarGAN** on https://github.com/sonynka/StarGAN.

# Usage

