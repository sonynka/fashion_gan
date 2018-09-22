# FashionGAN Search

This project was created as part of my master thesis to evaluate the possibilities of generative adversarial networks to improve image retrieval.

# Prerequisites
## Data
All data neccessary for running this project can be download by running the following script:

```bash
cd data && download_data.sh
```

### Images
The dataset used for this project is a scraped dataset of fashion product images and their attributes. The dataset was created by scraping several online fashion stores. The scraper project can be found here: https://github.com/sonynka/fashion_scraper. 

The models used in this project were trained on about 15.000 images of the category dresses from the original scraped dataset. Each of these product images also has a corresponding image of a model wearing the product.

### Features
In order to trigger a similarity search, all the product and model images from the dataset have various pre-calculated feature vectors as wel..

### Models
The models used for generation of new images were trained using several GANs repositories:
- **Pix2Pix** and **CycleGAN:** https://github.com/sonynka/pytorch-CycleGAN-and-pix2pix
- **StarGAN:** https://github.com/sonynka/StarGAN

# Usage

