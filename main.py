import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.models import vgg16_bn
import torchvision.transforms as transforms

from gradcam import gradcam

METHODS = ('Grad-Cam',)
# Adjust `INPUT_SIZE` and `NORMALIZE` to your own model
INPUT_SIZE = 224
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--index',
                        help='Class index. If `idx` is None, the index of the '
                             'maximum output will be used.',
                        type=int, default=None)
    parser.add_argument('--method',
                        help='Visual explanation method.',
                        type=str, choices=METHODS, default='Grad-Cam')
    parser.add_argument('image',
                        help='Path of input image.',
                        type=str)
    args = parser.parse_args()

    with open(args.image, 'rb') as f:
        image = Image.open(f).convert('RGB')
    resize = transforms.Resize((int(INPUT_SIZE), int(INPUT_SIZE)))
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE,
    ])
    image = resize(image)
    image_tensor = transform2(image)

    model = vgg16_bn(pretrained=True)
    model.eval()

    cam = gradcam(model, 'features', image_tensor, args.index)

    # Visualization
    cam = 1 - cam
    cam_heatmap = np.array(cv2.applyColorMap(np.uint8(255 * cam),
                                             cv2.COLORMAP_JET))
    cam_heatmap = cam_heatmap / 255
    image = np.array(image) / 255
    plt.imshow(0.6 * image + 0.4 * cam_heatmap)
    plt.show()


if __name__ == '__main__':
    main()
