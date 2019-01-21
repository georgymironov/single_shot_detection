import cv2

from . import dataset_utils


class VideoViewer(object):
    def __init__(self, path, predictor):
        self.path = path
        self.predictor = predictor

    def run(self):
        cap = cv2.VideoCapture(self.path.replace('file://', ''))
        cv2.namedWindow('image')

        while cap.isOpened():
            _, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ratio_w = rgb.shape[1] / self.predictor.resize.size[0]
            ratio_h = rgb.shape[0] / self.predictor.resize.size[1]

            prediction = self.predictor.predict_single(rgb)
            prediction[..., [0, 2]] *= ratio_w
            prediction[..., [1, 3]] *= ratio_h

            if dataset_utils.display(rgb, prediction) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
