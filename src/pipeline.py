from src.image_utils import plot_img, plot_img_row
from PIL import Image
from src.gans import Modifier


class FashionGANApp():
    """
    App that allows user to modify a dress image and search for similar
    products via console input.
    """

    def __init__(self, modifier: Modifier, dress_search, model_search,
                 num_imgs=10, metric='l1'):
        """
        :param modifier: GAN modifier object to generate modified images
        :param dress_search: Search or CombinedSearch object for products
        :param model_search: Search or CombinedSearch object for model images
        :param num_imgs: Number of similar images to retrieve
        """
        self._modifier = modifier
        self._dress_search = dress_search
        self._model_search = model_search
        self._num_similar_imgs = num_imgs
        self._search_metric = metric

    def start(self, input_img: Image):
        """
        Start the application - modify input image with console input.
        :param input_img: image to modify
        """

        product_img = input_img
        plot_img(product_img)

        # SHAPE MODIFICATION
        product_img = self._shape_modification(product_img)

        # PATTERN MODIFICATION
        product_img = self._pattern_modification(product_img)

        # MODEL IMAGE
        model_img = self._generate_model_image(product_img)
        mod_sim_imgs = self._search_models(model_img)

        # SEARCH
        prod_sim_imgs = self._search_products(product_img)

        # BEST IMAGE
        img_idx = self._select_best_image()

        # CONTINUE
        if img_idx != '':
            best_img = Image.open(prod_sim_imgs[int(img_idx)])
            self.start(best_img)

    def _shape_modification(self, img):
        self._print_title('SHAPE MODIFICATION')
        shape_labels = self._modifier.get_shape_labels()
        attr = self._ask_user_input(list(shape_labels.keys()), skip_option=True)

        if attr != '':
            value = self._ask_user_input(shape_labels[attr], skip_option=False)
            img = self._modifier.modify_shape(img, attr, value)
            plot_img(img)

        return img

    def _pattern_modification(self, img):
        self._print_title('PATTERN MODIFICATION')
        pattern_labels = self._modifier.get_pattern_labels()
        attr = self._ask_user_input(list(pattern_labels.keys()),
                                    skip_option=True)

        if attr != '':
            value = self._ask_user_input(pattern_labels[attr],
                                         skip_option=False)
            img = self._modifier.modify_pattern(img, attr, value)
            plot_img(img)

        return img

    def _generate_model_image(self, img):
        self._print_title('MODEL IMAGE')
        model_img = self._modifier.product_to_model(img)
        plot_img(model_img)

        return model_img

    def _search_products(self, img):
        self._print_title('SIMILAR PRODUCTS FROM PRODUCT SEARCH')
        prod_sim_imgs = self._dress_search.get_similar_images(
            img, self._num_similar_imgs, metric=self._search_metric)
        plot_img_row([Image.open(i) for i in prod_sim_imgs],
                     img_labels=range(self._num_similar_imgs))

        return prod_sim_imgs

    def _search_models(self, img):
        self._print_title('MODEL SEARCH')
        mod_sim_imgs = self._model_search.get_similar_images(
            img, num_imgs=self._num_similar_imgs, metric=self._search_metric)

        plot_img_row([Image.open(i) for i in mod_sim_imgs])

        return mod_sim_imgs

    def _select_best_image(self):
        self._print_title("SELECT BEST IMAGE")
        img_options = list(map(str, range(self._num_similar_imgs)))
        img_idx = self._ask_user_input(img_options, skip_option=True)

        return img_idx

    def _ask_user_input(self, options, skip_option=True):
        while True:
            print("Choose from the following: {}".format(options))
            if skip_option:
                print("or press ENTER to skip")
                options.append('')
            attr = input()

            if attr not in options:
                print("Invalid input, try again.")
            else:
                print()
                return attr

    @staticmethod
    def _print_title(title):
        print(title)
        print('-' * 30)


def main():
    from PIL import Image
    import os

    from src.gans import Modifier
    from src.features import AkiwiFeatureGenerator, ResnetFeatureGenerator
    from src.search import Search, CombinedSearch

    import warnings
    warnings.filterwarnings('ignore')

    folder_gens = {'akiwi_50': AkiwiFeatureGenerator(50),
                   'resnet': ResnetFeatureGenerator()}

    dress_imgs = '../data/images/fashion/dresses/'
    model_imgs = '../data/images/fashion_models/dresses_clustered/'

    dress_feats = '../data/features/fashion/dresses/'
    model_feats = '../data/features/fashion_models/dresses/'

    dress_search = {}
    for dir_name, gen in folder_gens.items():
        dress_search[dir_name] = Search(dress_imgs,
                                        os.path.join(dress_feats, dir_name),
                                        gen)

    model_search = {}
    for dir_name, gen in folder_gens.items():
        model_search[dir_name] = Search(model_imgs,
                                        os.path.join(model_feats, dir_name),
                                        gen)

    # combined search
    dress_resnet50 = CombinedSearch(
        [dress_search['akiwi_50'], dress_search['resnet']], factors=[2, 1])
    model_resnet50 = CombinedSearch(
        [model_search['akiwi_50'], model_search['resnet']], factors=[2, 1])

    modifier = Modifier('../data/models/')
    app = FashionGANApp(modifier, dress_resnet50, model_resnet50)

    test_img = Image.open('../data/images/fashion/dresses/1DR21C07A-Q11.jpg')
    app.start(test_img)

    print('done')


if __name__ == '__main__':
    main()