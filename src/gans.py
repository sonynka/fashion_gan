from networks import stargan, pix2pix, cyclegan
from torchvision import transforms
import torch
from PIL import Image
import os


class Modifier():

    def __init__(self, models_root):
        self._shape_modifier = _StarGANModifier(os.path.join(models_root, 'stargan'))
        self._pattern_modifier = _CycleGANModifier(os.path.join(models_root, 'cyclegan'))
        self._model_generator = _Pix2PixModifier(os.path.join(models_root, 'pix2pix_models.pth'))

    def modify_shape(self, image: Image, attribute: str, value: str):
        return self._shape_modifier.modify_image(image, attribute, value)

    def modify_pattern(self, image: Image, attribute: str, value: str):
        return self._pattern_modifier.modify_image(image, attribute, value)

    def product_to_model(self, image: Image):
        return self._model_generator.generate_image(image)

    def get_shape_labels(self):
        return self._shape_modifier.LABELS

    def get_pattern_labels(self):
        return self._pattern_modifier.LABELS


class _BaseModifier():
    def __init__(self, img_size):

        self.TRANSFORMS = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    @staticmethod
    def denorm_tensor(img_tensor):

        img_d = (img_tensor + 1) / 2
        img_d = img_d.clamp_(0, 1)
        img_d = img_d.data.mul(255).clamp(0, 255).byte()
        img_d = img_d.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(img_d)


class _StarGANModifier(_BaseModifier):

    LABELS = {
        'sleeve_length':    ['3/4', 'long', 'short', 'sleeveless'],
        'fit':              ['loose', 'normal', 'tight'],
        'neckline':         ['round', 'v', 'wide'],
        'length':           ['short', 'knee', 'long']
    }

    IMAGE_SIZE = 128

    def __init__(self, G_path_root):

        self.G_models = {}

        for attr, label_map in self.LABELS.items():
            num_classes = len(self.LABELS[attr])

            try:
                G_path = os.path.join(G_path_root, attr + '.pth')
                G = stargan.Generator(c_dim=num_classes)
                G.load_state_dict(torch.load(G_path, map_location='cpu'))
                self.G_models[attr] = G
            except:
                print("Couldn't find model", G_path)

        super().__init__(self.IMAGE_SIZE)

    def modify_image(self, image: Image, attribute: str, value: str):
        assert attribute in self.LABELS.keys()
        assert value in self.LABELS[attribute]

        img_tensor = self.TRANSFORMS(image).unsqueeze(0)
        img_label = self.get_label(attribute, value)
        label_tensor = torch.FloatTensor(img_label).unsqueeze(0)

        G = self.G_models[attribute]
        fake_tensor = G(img_tensor, label_tensor).squeeze(0)
        fake_img = self.denorm_tensor(fake_tensor)

        return fake_img

    def get_label(self, attr, value):
        attr_len = len(self.LABELS[attr])
        label_idx = self.LABELS[attr].index(value)

        label = [0] * attr_len
        label[label_idx] = 1

        return label


class _CycleGANModifier(_BaseModifier):

    LABELS = {
        'floral': ['add', 'remove'],
        'stripes': ['add', 'remove']
    }

    IMAGE_SIZE = 256

    def __init__(self, G_path_root):

        self.G_models = {}

        for attr, values in self.LABELS.items():
            self.G_models[attr] = {}

            for value in values:
                try:
                    G_path = os.path.join(
                        G_path_root, '{}_{}.pth'.format(attr, value))
                    G = cyclegan.Generator()
                    G.load_state_dict(torch.load(G_path, map_location='cpu'))

                    self.G_models[attr][value] = G
                except:
                    print("Couldn't find model", G_path)

        super().__init__(self.IMAGE_SIZE)

    def modify_image(self, image: Image, attribute: str, value: str):
        assert attribute in self.LABELS.keys()
        assert value in self.LABELS[attribute]

        img_tensor = self.TRANSFORMS(image).unsqueeze(0)
        fake_tensor = self.G_models[attribute][value](img_tensor).squeeze(0)
        fake_img = self.denorm_tensor(fake_tensor)

        return fake_img


class _Pix2PixModifier(_BaseModifier):

    IMAGE_SIZE = 256

    def __init__(self, G_path):

        self.G = pix2pix.Generator()
        self.G.load_state_dict(torch.load(G_path, map_location='cpu'))

        super().__init__(self.IMAGE_SIZE)

    def generate_image(self, image: Image):
        img_tensor = self.TRANSFORMS(image).unsqueeze(0)
        fake_tensor = self.G(img_tensor).squeeze(0)
        fake_img = self.denorm_tensor(fake_tensor)

        return fake_img


def main():
    modifier = Modifier('../data/models/')

    test_img = Image.open('../data/images/test_images/dresses/605423287.jpg')
    mod_img = modifier.modify_shape(test_img, 'sleeve_length', 'long')
    print('done')

if __name__ == '__main__':
    main()