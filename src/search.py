import numpy as np
import os
import glob
from PIL import Image
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from src import image_utils
from src.features import ResnetFeatureGenerator, AkiwiFeatureGenerator


class Search():
    """
    Loads all features from a folder and searches best images by comparing
    the distances between the features.
    """

    def __init__(self, images_root, features_root, feature_generator=None):
        """
        :param features_root: path to folder containing .npy features
        :param images_root:   path to folder containg images with same names as
                              in features_root
        :param feature_generator: generator to create features for new images
        """

        if not feature_generator:
            feature_generator = ResnetFeatureGenerator()

        self.features_root = features_root
        self.feature_generator = feature_generator
        self.images_root = images_root

        self.feature_names, self.features = self._load_features()
        self.scaler = self._scale_features()

    def _load_features(self):
        """
        Load features from the feature root into a numpy array
        """

        print('Loading features from:', self.features_root)
        feat_files = glob.glob(os.path.join(self.features_root, '*.npy'))

        if len(feat_files) == 0:
            raise ValueError(
                'Features root {} is empty.'.format(self.features_root))

        feat_files = sorted(feat_files)
        feat_names = [os.path.basename(f).rsplit('.', 1)[0]
                      for f in feat_files]
        features = np.array([np.load(f) for f in feat_files])

        return feat_names, features

    def _scale_features(self):
        """
        Scale all loaded features to mean=0 and std=1
        """
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

        return scaler

    def _get_img_feature(self, img):
        img = image_utils.process_image(img)

        img_feature = self.feature_generator.get_feature(img)
        img_feature = img_feature.reshape(1, -1)
        img_feature = self.scaler.transform(img_feature)

        return img_feature

    def _search_similar_images(self, img, num_imgs, metric):
        """
        Calculate pairwise distances from the given image to the loaded
        features and return the image paths with the smallest distances
        """

        img_feature = self._get_img_feature(img)

        dist = pairwise_distances(img_feature, self.features, metric=metric)
        best_img_idxs = np.argsort(dist)[0].tolist()[:num_imgs]
        best_img_dist = dist[0][best_img_idxs]

        best_img_paths = [
            os.path.join(self.images_root, self.feature_names[i] + '.jpg')
            for i in best_img_idxs]

        return best_img_paths, best_img_dist

    def get_similar_images_with_distances(self, img: Image, num_imgs=8, metric='l1'):
        """
        Retrieve similar images and their distances to the given image compared
        with the given metric
        :param img:         image for which to find similar images
        :param num_imgs:    number of similar images to retrieve
        :param metric:      distance metric to use when comparing features ['l1','l2']
        :return:            list of best image paths and list of their distances
                            to the original image
        """
        return self._search_similar_images(img, num_imgs, metric)

    def get_similar_images(self, img: Image, num_imgs=8, metric='l1'):
        """
        Retrieve similar images to the given image compared with the given metric
        :param img:         image for which to find similar images
        :param num_imgs:    number of similar images to retrieve
        :param metric:      distance metric to use when comparing features ['l1','l2']
        :return:            list of best image paths
        """

        best_img_paths, _ = self._search_similar_images(img, num_imgs, metric)
        return best_img_paths


class CombinedSearch():
    """
    Combines a list of Search classes to enable search with combined features
    """

    def __init__(self, search_list, factors=None):
        """
        :param search_list: list of Search objects
        :param factors:     list of factors with which to multiply the
                            respective Search features
        """

        # take all features as equal
        if not factors:
            factors = [1] * len(search_list)

        self.factors = factors
        self.search_list = search_list
        self.features = []

        for search, factor in zip(search_list, factors):
            self.features.append(factor * search.features)
        self.features = np.concatenate(self.features, axis=1)

    def _get_img_feature(self, img):
        img = image_utils.process_image(img)

        img_feature = []
        for search, factor in zip(self.search_list, self.factors):
            feature = search.feature_generator.get_feature(img)
            feature = feature.reshape(1, -1)
            feature = search.scaler.transform(feature)
            img_feature.append(factor * feature.squeeze())

        img_feature = np.concatenate(img_feature)
        img_feature = img_feature.reshape(1, -1)

        return img_feature

    def _search_similar_images(self, img, num_imgs, metric):
        """
        Calculate pairwise distances from the given image to the loaded
        features and return the image paths with the smallest distances
        """

        img_feature = self._get_img_feature(img)

        dist = pairwise_distances(img_feature, self.features, metric=metric)
        best_img_idxs = np.argsort(dist)[0].tolist()[:num_imgs]
        best_img_dist = dist[0][best_img_idxs]

        best_img_paths = [
            os.path.join(self.search_list[0].images_root,
                         self.search_list[0].feature_names[i] + '.jpg')
            for i in best_img_idxs]

        return best_img_paths, best_img_dist

    def get_similar_images_with_distances(self, img: Image, num_imgs=8, metric='l1'):
        """
        Retrieve similar images and their distances to the given image compared
        with the given metric
        :param img:         image for which to find similar images
        :param num_imgs:    number of similar images to retrieve
        :param metric:      distance metric to use when comparing features ['l1','l2']
        :return:            list of best image paths and list of their distances
                            to the original image
        """
        return self._search_similar_images(img, num_imgs, metric)

    def get_similar_images(self, img: Image, num_imgs=8, metric='l1'):
        """
        Retrieve similar images to the given image compared with the given metric
        :param img:         image for which to find similar images
        :param num_imgs:    number of similar images to retrieve
        :param metric:      distance metric to use when comparing features ['l1','l2']
        :return:            list of best image paths
        """

        best_img_paths, _ = self._search_similar_images(img, num_imgs, metric)
        return best_img_paths


def main():
    print('done')


if __name__ == '__main__':
    main()