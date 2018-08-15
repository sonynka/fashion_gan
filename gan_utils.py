from models.stargan import Generator
from torchvision import transforms
import torch

from PIL import Image
import os

class StarGAN_generator():

    LABELS = {
        'sleeve_length': {
            'sleeveless': [0, 0, 0, 1],
            'short': [0, 0, 1, 0],
            '3/4': [1, 0, 0, 0],
            'long': [0, 1, 0, 0]}
    }

    TRANSFORMS = transforms.Compose([
        transforms.Resize(128, interpolation=Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __init__(self, G_path_root):

        self.G_models = {}

        for attr, label_map in self.LABELS.items():
            num_classes = len(self.LABELS[attr])

            G_path = os.path.join(G_path_root, attr + '.pth')
            G = Generator(c_dim=num_classes)
            G.load_state_dict(torch.load(G_path, map_location='cpu'))

            self.G_models[attr] = G

    def generate_image(self, image: Image, attr: str, value: str):

        img_tensor = self.TRANSFORMS(image).unsqueeze(0)

        img_label = self.LABELS[attr][value]
        label_tensor = torch.FloatTensor(img_label).unsqueeze(0)

        G = self.G_models[attr]
        fake_tensor = G(img_tensor, label_tensor).squeeze(0)
        fake_img = self.denorm_tensor(fake_tensor)

        return fake_img

    def denorm_tensor(self, img_tensor):

        img_d = (img_tensor + 1) / 2
        img_d = img_d.clamp_(0, 1)
        img_d = img_d.data.mul(255).clamp(0, 255).byte()
        img_d = img_d.permute(1, 2, 0).cpu().numpy()

        return Image.fromarray(img_d)