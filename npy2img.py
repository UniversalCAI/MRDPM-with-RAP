import numpy as np
import matplotlib.pyplot as plt
import os


def npy2img(npy_path, img_path):

    for root, _, fnames in sorted(os.walk(npy_path)):
        for f in fnames:
            if f.endswith('.npy'):
                if "Process" not in f:
                    npy_path = os.path.join(root, f)
                    img_path = os.path.join(root, f.split('.')[0] + '.png')
                    img = np.load(npy_path).clip(0, 1)*255
                    # turn to img
                    img = img.astype(np.uint8)
                    plt.imsave(img_path, img, cmap='gray')
