from networks import stargan, pix2pix
from torchvision import transforms
import torch

from PIL import Image
import os

class StarGAN_generator():

    LABELS = {
        'sleeve_length':    ['3/4', 'long', 'short', '`sleeveless'],
        'pattern':          ['floral', 'lace', 'polkadots', 'print',
                             'stripes', 'unicolors']
    }

    TRANSFORMS = transforms.Compose([
        transforms.Resize(128, interpolation=Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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

    def generate_image(self, image: Image, attr: str, value: str):

        img_tensor = self.TRANSFORMS(image).unsqueeze(0)

        img_label = self.get_label(attr, value)
        label_tensor = torch.FloatTensor(img_label).unsqueeze(0)

        G = self.G_models[attr]
        fake_tensor = G(img_tensor, label_tensor).squeeze(0)
        fake_img = self.denorm_tensor(fake_tensor)

        return fake_img

    def get_label(self, attr, value):
        assert attr in self.LABELS.keys()
        assert value in self.LABELS[attr]

        attr_len = len(self.LABELS[attr])
        label_idx = self.LABELS[attr].index(value)

        label = [0] * attr_len
        label[label_idx] = 1

        return(label)

    def denorm_tensor(self, img_tensor):

        img_d = (img_tensor + 1) / 2
        img_d = img_d.clamp_(0, 1)
        img_d = img_d.data.mul(255).clamp(0, 255).byte()
        img_d = img_d.permute(1, 2, 0).cpu().numpy()

        return Image.fromarray(img_d)


class Pix2PixGenerator():

    TRANSFORMS = transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __init__(self, G_path):

        self.G = pix2pix.Generator()
        self.G.load_state_dict(torch.load(G_path, map_location='cpu'))

    def generate_image(self, image: Image):
        img_tensor = self.TRANSFORMS(image).unsqueeze(0)
        fake_tensor = self.G(img_tensor).squeeze(0)
        fake_img = self.denorm_tensor(fake_tensor)

        return fake_img

    def denorm_tensor(self, img_tensor):
        img_d = (img_tensor + 1) / 2
        img_d = img_d.clamp_(0, 1)
        img_d = img_d.data.mul(255).clamp(0, 255).byte()
        img_d = img_d.permute(1, 2, 0).cpu().numpy()

        return Image.fromarray(img_d)



def main():
    s = StarGAN_generator('./models/stargan/')
    test_img_path = './test_images/dresses_sample/IV321C031-Q11.jpg'
    test_img = Image.open(test_img_path)
    fake_img = s.generate_image(test_img, 'sleeve_length', 'long')

    print('done')

if __name__ == '__main__':
    main()