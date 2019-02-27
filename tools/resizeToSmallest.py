import glob
import sys

from bs4 import BeautifulSoup
import cv2
from joblib import Parallel, delayed


def resize(path, widht, height):
    img = cv2.imread(path)
    h, w, d = img.shape

    ratio = min(h / height, w / widht)
    w /= ratio
    h /= ratio
    w = int(w)
    h = int(h)

    img = cv2.resize(img, (w, h))
    cv2.imwrite(path, img)

    with open(path.replace('.jpg', '.xml'), 'r') as f:
        soup = BeautifulSoup(f, 'lxml-xml')

    if soup.find('width') is None or soup.find('height') is None:
        print(path)
        return

    soup.find('width').string = str(w)
    soup.find('height').string = str(h)

    for attr in ['xmin', 'xmax', 'ymin', 'ymax']:
        for x in soup.find_all(attr):
            if x is not None:
                new_value = int(float(x.string) / ratio)
                assert new_value < w and new_value < h
                x.string = str(new_value)

    with open(path.replace('.jpg', '.xml'), 'w') as f:
        f.write(str(soup))

if __name__ == '__main__':
    path, width, height = sys.argv[1:4]
    width = int(width)
    height = int(height)
    imgs = glob.glob(f'{path}/**/*.jpg', recursive=True)
    r = Parallel(n_jobs=8, verbose=10)(delayed(resize)(x, width, height) for x in imgs)
