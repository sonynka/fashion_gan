#!/usr/bin/env bash
echo "Downloading images..."
curl -Lo images.zip https://s3.eu-central-1.amazonaws.com/fashion-gan/images.zip
echo "Downloading features..."
curl -Lo features.zip https://s3.eu-central-1.amazonaws.com/fashion-gan/features.zip
echo "Downloading models..."
curl -Lo models.zip https://s3.eu-central-1.amazonaws.com/fashion-gan/models.zip
echo "Downloading clustering data..."
curl -Lo clustering.zip https://s3.eu-central-1.amazonaws.com/fashion-gan/clustering.zip

echo "Unzipping files..."
unzip -q images.zip
unzip -q features.zip
unzip -q models.zip
unzip -q clustering.zip

rm *.zip
