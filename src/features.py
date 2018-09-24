import numpy as np
import os
import io
import requests
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms


class ResnetFeatureGenerator():
    """
    Loads a ResNet152 model to generate feature vectors from the last hidden
    layer of the network. If no model path is provided, the pretrained ImageNet
    model from PyTorch is loaded. Otherwise, the model's weights are loaded
    from the model path.
    """

    _DATA_TRANSFORMS = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self, retrained_model_path=None):
        """
        :param retrained_model_path: path to a retrained resnet152 model
        (if not provided, the original ImageNet pretrained model from PyTorch
        is loaded)
        """

        self.model_path = retrained_model_path
        self.model = self._load_resnet152()

    def get_feature(self, img: Image):
        feature = self.model(self._DATA_TRANSFORMS(img).unsqueeze(0))
        feature = feature.squeeze().data.numpy()

        return feature

    def _load_resnet152(self):
        """Loads original ResNet152 model with pretrained weights on ImageNet
        dataset. If a model_path is provided, it loads the weights from the
        model_path. Returns the model excluding the last layer.
        """

        model = torchvision.models.resnet152(pretrained=True)

        if self.model_path:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 7)
            model.load_state_dict(
                torch.load(self.model_path, map_location='cpu'))

        # only load layers until the last hidden layer to extract features
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)

        return model


class AkiwiFeatureGenerator():
    """
    Queries the Akiwi API to get feature vectors for an image.
    """

    _SIZE_URLS = {
        64:     ['http://akiwi.eu/mxnet/feature/'],
        50:     ['http://akiwi.eu/feature/fv50/'],
        114:    ['http://akiwi.eu/mxnet/feature/',
                 'http://akiwi.eu/feature/fv50/']
    }

    def __init__(self, feature_size):
        """
        Based on the feature size, queries the corresponding API endpoint.
        Feature size 114 are sizes 64 and 50 appended.
        :param feature_size: the size of the feature vector to download
        """
        self.urls = self._SIZE_URLS[feature_size]

    def get_feature(self, img: Image):

        features = []
        for url in self.urls:
            f = self._get_url_feature(url, img)
            features.append(f)

        feature = np.concatenate(features)
        return feature

    @staticmethod
    def _get_url_feature(url, img: Image):
        """
        Query the Akiwi API to get an image feature vector
        :param url: URL of the API to query
        :param img: image for which to get feature vector
        :return: feature numpy array
        """

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        files = {'file': ('img.jpg', img_bytes.getvalue(), 'image/jpeg')}
        response = requests.post(url, files=files, timeout=10)

        retries = 0
        while (response.status_code != 200) & (retries < 3):
            retries += 1
            response = requests.post(url, files=files, timeout=10)

        if response.status_code == 200:
            response_feature = response.content
            feature = np.frombuffer(response_feature, dtype=np.uint8)
            return feature

        print("Couldn't get feature. Response: ", response)


def download_feature_vectors(files, save_dir, feature_generator=None):
    """
    Downloads feature vectors for a list of files.
    :param files: list of files
    :param save_dir: directory to save feature vectors
    :param feature_generator: feature generator to create features
    """

    if not feature_generator:
        feature_generator = AkiwiFeatureGenerator(114)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, file in enumerate(files):
        if idx % 100 == 0:
            print('Downloaded {} / {}'.format(idx, len(files)))

        # name feature with the same basename as file
        save_path = os.path.join(save_dir, os.path.basename(file).split('.jpg')[0] + '.npy')
        if os.path.exists(save_path):
            continue

        feature = feature_generator.get_feature(Image.open(file))
        np.save(save_path, feature)


def main():
    print('done')


if __name__ == '__main__':
    main()