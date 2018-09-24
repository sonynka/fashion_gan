from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def get_edges(img_path, img_size, crop=False):
    """ Get black-white canny edge detection for the image """

    img = Image.open(img_path)
    if crop:
        w, h = img.size
        img = img.crop((70, 0, w - 70, h))
    img = img.resize(img_size, Image.ANTIALIAS)
    img = img.filter(ImageFilter.FIND_EDGES)
    img = img.convert('L')
    img_arr = np.array(img)
    img.close()

    return img_arr


def get_mask(img_path, img_size, crop=False):

    img = threshold_mask(img_path)

    if crop:
        w, h = img.size
        img = img.crop((70, 0, w - 70, h))
    img = img.resize(img_size)
    img_arr = np.array(img)
    img.close()

    return img_arr


def threshold_mask(img_path, thresh=200):
    """
    Get image threshold mask.
    """

    # Read image
    im_in = cv2.imread(img_path)

    # grayscale and blur
    im_in = cv2.cvtColor(im_in, cv2.COLOR_BGR2GRAY)
    im_in = cv2.GaussianBlur(im_in, (5, 5), 0)

    # get threshold mask
    _, thresh_mask = cv2.threshold(im_in, thresh, 255, cv2.THRESH_BINARY)
    im_masked = cv2.bitwise_not(thresh_mask)

    # fill contours of threshold mask
    _, contours, _ = cv2.findContours(im_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(im_masked, [cnt], 0, 255, -1)

    # smooth and dilate
    smooth = cv2.GaussianBlur(im_masked, (9, 9), 0)
    _, thresh_mask2 = cv2.threshold(smooth, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    im_out = cv2.dilate(thresh_mask2, kernel, iterations=1)

    return Image.fromarray(im_out)


def plot_img_row(images, img_labels=None):
    fig, axarr = plt.subplots(nrows=1, ncols=len(images),
                              figsize=(len(images) * 3, 4))

    for i, img in enumerate(images):
        ax = axarr[i]
        img = img.resize([256, 256])
        img = img.crop((40, 0, 216, 256))
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.keys():
            ax.spines[spine].set_visible(False)

        if img_labels is not None:
            ax.set_title(img_labels[i])

    plt.show()


def plot_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    return img