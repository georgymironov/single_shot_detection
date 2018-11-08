from skimage.draw import line
import skimage.io
import torch


def display(img, target):
    if isinstance(img, torch.Tensor):
        img = img.numpy().transpose((1, 2, 0))
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    for box in target:
        xmin = int(max(0, box[0]))
        ymin = int(max(0, box[1]))
        xmax = int(min(img.shape[1] - 1, box[2]))
        ymax = int(min(img.shape[0] - 1, box[3]))
        dots = [(ymin, xmin), (ymax, xmin), (ymax, xmax), (ymin, xmax)]
        for i, current in enumerate(dots):
            next_ = dots[(i + 1) % len(dots)]
            img[line(current[0], current[1], next_[0], next_[1])] = (0, 255, 0)

    skimage.io.imshow(img)
    skimage.io.show()
