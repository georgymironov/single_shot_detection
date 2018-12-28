from torch.utils.data import Dataset

from bf.utils import dataset_utils


class DetectionDataset(Dataset):
    @staticmethod
    def collate(batch):
        return dataset_utils.collate_detections(batch)

    def display(self, index):
        dataset_utils.display(*self[index])
