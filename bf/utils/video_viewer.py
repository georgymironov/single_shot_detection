import glob
import mimetypes
import os

import cv2

from . import dataset_utils


def _is_video(f):
    t = mimetypes.guess_type(f)[0]
    if isinstance(t, str):
        return t.split('/')[0] == 'video'
    return False

class VideoViewer(object):
    def __init__(self, path, predictor):
        path = path.replace('file://', '')

        if os.path.isdir(path):
            self.paths = [x for x in glob.glob(os.path.join(path, '**'), recursive=True) if _is_video(x)]
        else:
            self.paths = [path]

        self.predictor = predictor

    def run(self):
        stop = False
        for path in self.paths:
            print(path)

            cap = cv2.VideoCapture(path)
            cv2.namedWindow('image')

            while cap.isOpened():
                _, frame = cap.read()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                prediction = self.predictor.predict_single(rgb)

                result = dataset_utils.display(rgb, prediction)

                if result & 0xFF == ord('Q'):
                    stop = True
                    break

                if result & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            if stop:
                break
