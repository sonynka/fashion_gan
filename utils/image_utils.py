from PIL import Image, ImageOps


def pad_image(img):
    width, height = img.size

    max_size = max(width, height)

    pad_height = max_size - height
    pad_width = max_size - width

    padding = (pad_width // 2,
               pad_height // 2,
               pad_width - (pad_width // 2),
               pad_height - (pad_height // 2))

    padded_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
    return padded_img


def remove_alpha(img):
    if img.mode == 'RGBA':
        img.load()  # required for png.split()
        image_jpeg = Image.new("RGB", img.size, (255, 255, 255))
        image_jpeg.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = image_jpeg

    return img


def resize_image(img, size):
    return img.resize(size, Image.ANTIALIAS)


def process_image(img: Image, size=None):
    if not size:
        size = [256, 256]

    img = remove_alpha(img)
    img = pad_image(img)
    img = resize_image(img, size=size)

    return img