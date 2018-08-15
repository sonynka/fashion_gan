import numpy as np
import os
import glob
import io
import requests
from PIL import Image
import torchvision
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import pairwise_distances


class Search():

    def __init__(self, images_root, features_root,
                 feature_generator=None):
        """
        Loads all the given features and searches best images by comparing
        the L2 distances.
        :param features_root:   path to folder containing .npy features
        :param images_root:     path to folder containg images with same names
                                as in features_root
        """

        if feature_generator is None:
            feature_generator = ResnetFeatureGenerator()

        self.feature_generator = feature_generator
        self.feat_dict = self.load_img_features(features_root)
        self.images_root = images_root

        self.feat_names = list(self.feat_dict.keys())
        self.features = np.array(list(self.feat_dict.values()))

    def load_img_features(self, features_root):

        files = glob.glob(os.path.join(features_root, '*.npy'))
        feat_dict = {os.path.basename(f).rsplit('.', 1)[0]: np.load(f)
                     for f in files}
        return feat_dict

    def load_img_paths(self, images_root):
        img_files = glob.glob(images_root)
        return img_files

    def get_similar_images(self, img, num_imgs=8):

        img_feature = self.feature_generator.get_image_feature(img)
        dist = pairwise_distances(img_feature.reshape(1, -1), self.features)
        best_img_idxs = np.argsort(dist)[0].tolist()[:num_imgs]

        best_img_names = [self.feat_names[i] for i in best_img_idxs]
        best_imgs = [Image.open(os.path.join(self.images_root, n + '.jpg'))
                     for n in best_img_names]

        return best_imgs

    @staticmethod
    def get_img_id_from_path(path):
        return os.path.basename(path).rsplit('.', 1)[0]


class ResnetFeatureGenerator():

    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self):
        self.model = self.load_resnet152_original()

    def get_image_feature(self, img: Image):

        feature = self.model(self.data_transforms(img).unsqueeze(0))
        feature = feature.squeeze().data.numpy()

        return feature

    @staticmethod
    def load_resnet152_original():
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)

        return model


class AkiwiFeatureGenerator():

    SIZE_URLS = {
        64:     ['http://akiwi.eu/mxnet/feature/'],
        50:     ['http://akiwi.eu/feature/fv50/'],
        114:    ['http://akiwi.eu/mxnet/feature/',
                 'http://akiwi.eu/feature/fv50/']
    }

    def __init__(self, feature_size):
        self.urls = self.SIZE_URLS[feature_size]

    def get_image_feature(self, img: Image):

        features = []
        for url in self.urls:
            f = self.__get_url_feature(url, img)
            features.append(f)

        feature = np.concatenate(features)
        return(feature)

    def __get_url_feature(self, url, img):

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        files = {'file': ('img.jpg', img_bytes.getvalue(), 'image/jpeg')}
        response = requests.post(url, files=files, timeout=10)

        if response.status_code == 200:
            response_feature = response.content
            feature = np.frombuffer(response_feature, dtype=np.uint8)
            return feature
        else:
            print("Couldn't get feature. Response: ", response)