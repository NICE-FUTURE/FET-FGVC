from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import  Image
import numpy as np
import os


class BatchDataset(Dataset):

    def __init__(self, root_dir, stage, txt_dir, transform=None, namelist=None):
        self.transform = transform
        if stage == "train":
            self.txt_path = os.path.join(txt_dir, "train.txt")
        elif stage == "val":
            self.txt_path = os.path.join(txt_dir, "val.txt")
        self.bird_names_path = os.path.join(txt_dir, "classes.txt")
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))

        if namelist:
            self.imglist = namelist
        else:
            with open(self.txt_path, 'r', encoding="utf-8") as fid:
                self.imglist = fid.readlines()
        self.label_ids = []
        cnt = 0
        with open(self.bird_names_path, "r", encoding="utf-8") as f:
            for line in f:
                self.label_ids.append(cnt)
                cnt += 1
        self.n_classes = len(self.label_ids)
        self.all_labels = np.array([int(line.strip().split(",")[1]) for line in self.imglist])

    def __getitem__(self, index):
        image_path, label = self.imglist[index].strip().split(",")
        image_path = os.path.join(self.root_dir, image_path)
        filename = os.path.basename(image_path)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(label)

        return image, label, filename

    def __len__(self):
        return len(self.imglist)


class BalancedBatchSampler(BatchSampler):
    """
    sampler
    """
    def __init__(self, dataset, n_classes, n_samples):
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.all_labels = dataset.all_labels
        self.label_ids = dataset.label_ids
        self.label_indices = {label: np.where(self.all_labels == label)[0] for label in self.label_ids}
        for key in self.label_ids:
            np.random.shuffle(self.label_indices[key])
        self.used_label_indices_count = {label: 0 for label in self.label_ids}
        self.count = 0
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            choosed_labels = np.random.choice(self.label_ids, self.n_classes, replace=False)
            indices = []
            for label in choosed_labels:
                start = self.used_label_indices_count[label]
                indices.extend(self.label_indices[label][start:start + self.n_samples])
                self.used_label_indices_count[label] += self.n_samples
                if self.used_label_indices_count[label] + self.n_samples > len(self.label_indices[label]):
                    np.random.shuffle(self.label_indices[label])
                    self.used_label_indices_count[label] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
