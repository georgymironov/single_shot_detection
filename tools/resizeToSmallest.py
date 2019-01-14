import glob
import sys

from bs4 import BeautifulSoup
import cv2
from joblib import Parallel, delayed


def resize(path, widht, height):
    img = cv2.imread(path)
    h, w, d = img.shape

    r = min(h / height, w / widht)
    w /= r
    h /= r
    w = int(w)
    h = int(h)

    img = cv2.resize(img, (w, h))
    cv2.imwrite(path, img)

    with open(path.replace('.jpg', '.xml'), 'r') as f:
        s = BeautifulSoup(f, 'lxml-xml')

    if s.find('width') is None or s.find('height') is None:
        print(path)
        return

    s.find('width').string = str(w)
    s.find('height').string = str(h)

    for a in ['xmin', 'xmax', 'ymin', 'ymax']:
        for x in s.find_all(a):
            if x is not None:
                x.string = str(int(float(x.string) / r))

    with open(path.replace('.jpg', '.xml'), 'w') as f:
        f.write(str(s))

if __name__ == '__main__':
    path, width, height = sys.argv[1:4]
    width = int(width)
    height = int(height)
    imgs = glob.glob(f'{path}/**/*.jpg', recursive=True)
    r = Parallel(n_jobs=8, verbose=10)(delayed(resize)(x, width, height) for x in imgs)
