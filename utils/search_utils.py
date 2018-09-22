import numpy as np
import os
import glob
import io
import requests
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import image_utils


class CombinedSearch():

    def __init__(self, search_list, factors=None):
        """
        """
        if not factors:
            factors = [1, 1]

        self.factors = factors
        self.search_list = search_list
        self.features = []

        for search, factor in zip(search_list, factors):
            self.features.append(factor * search.features)

        self.features = np.concatenate(self.features, axis=1)

    def get_similar_images(self, img, num_imgs=8, metric='l1', get_distances=False):
        img = image_utils.process_image(img)

        img_feature = []
        for search, factor in zip(self.search_list, self.factors):
            feature = search.feature_generator.get_feature(img)
            feature = feature.reshape(1, -1)
            feature = search.scaler.transform(feature)
            img_feature.append(factor * feature.squeeze())

        img_feature = np.concatenate(img_feature)
        img_feature = img_feature.reshape(1, -1)

        dist = pairwise_distances(img_feature, self.features, metric=metric)
        best_img_idxs = np.argsort(dist)[0].tolist()[:num_imgs]
        best_img_dist = dist[0][best_img_idxs]

        best_img_paths = [os.path.join(self.search_list[0].images_root,
                                  self.search_list[0].feature_names[i] + '.jpg')
            for i in best_img_idxs]

        if get_distances:
            return best_img_paths, best_img_dist
        else:
            return best_img_paths

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

        if not feature_generator:
            feature_generator = ResnetFeatureGenerator()

        self.features_root = features_root
        self.feature_generator = feature_generator
        self.images_root = images_root

        self.feature_names, self.features = self.load_features()
        self.scaler = self.scale_features()

    def load_features(self):
        print('Loading features from:', self.features_root)
        feat_files = glob.glob(os.path.join(self.features_root, '*.npy'))

        if len(feat_files) == 0:
            raise ValueError('Features root is empty.')

        feat_files = sorted(feat_files)

        feat_names = [os.path.basename(f).rsplit('.', 1)[0]
                      for f in feat_files]
        features = np.array([np.load(f) for f in feat_files])

        return feat_names, features

    def scale_features(self):
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

        return scaler

    def get_similar_images(self, img: Image, num_imgs=8, metric='l1', get_distances=False):
        img = image_utils.process_image(img)

        img_feature = self.feature_generator.get_feature(img)
        img_feature = img_feature.reshape(1, -1)
        img_feature = self.scaler.transform(img_feature)

        dist = pairwise_distances(img_feature, self.features, metric=metric)
        best_img_idxs = np.argsort(dist)[0].tolist()[:num_imgs]
        best_img_dist = dist[0][best_img_idxs]

        best_img_paths = [os.path.join(self.images_root, self.feature_names[i] + '.jpg')
                     for i in best_img_idxs]

        if get_distances:
            return best_img_paths, best_img_dist
        else:
            return best_img_paths

class ResnetFeatureGenerator():

    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self, retrained_model_path=None):

        self.model_path = retrained_model_path
        self.model = self.load_resnet152()

    def get_feature(self, img: Image):
        feature = self.model(self.data_transforms(img).unsqueeze(0))
        feature = feature.squeeze().data.numpy()

        return feature

    def load_resnet152(self):
        model = torchvision.models.resnet152(pretrained=True)

        if self.model_path:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 7)
            model.load_state_dict(torch.load(self.model_path,
                                             map_location='cpu'))

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

    def get_feature(self, img: Image):

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


def download_feature_vectors(files, save_dir, feature_gen=None):
    if not feature_gen:
        feature_gen = AkiwiFeatureGenerator(114)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, file in enumerate(files):
        if idx % 100 == 0:
            print('Downloaded {} / {}'.format(idx, len(files)))

        save_path = os.path.join(save_dir, os.path.basename(file).split('.jpg')[0] + '.npy')
        if os.path.exists(save_path):
            continue

        feature = feature_gen.get_feature(Image.open(file))
        np.save(save_path, feature)

def main():
    product_imgs = '/Users/sonynka/HTW/MasterArbeit/data/fashion/dresses/'
    product_feats_root = '/Users/sonynka/HTW/MasterArbeit/Projects/fashion_designer/data/features/fashion/dresses/'

    folder_gens = {'akiwi_50': AkiwiFeatureGenerator(50),
                   'akiwi_64': AkiwiFeatureGenerator(64),
                   'akiwi_114': AkiwiFeatureGenerator(114),
                   'resnet': ResnetFeatureGenerator(),
                   'resnet_retrained': ResnetFeatureGenerator(
                       './models/resnet152_retrained.pth')
                   }

    searches = {}
    for dir_name, gen in folder_gens.items():
        searches[dir_name] = Search(product_imgs,
                                    os.path.join(product_feats_root, dir_name),
                                    gen)

    searches['akiwi_50'].get_similar_images(Image.open('/Users/sonynka/HTW/MasterArbeit/data/fashion/dresses/5713733606269.jpg'))
    print('done')

if __name__ == '__main__':
    main()