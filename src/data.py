import warnings
import torchvision.io as io
from torch.utils.data import Dataset
from glob import glob

class ClassifierDataset(Dataset):
    def __init__(self, ds_path, filetype=".jpg", transform=None):
        super(ClassifierDataset,self).__init__()
        data = []
        labels = []
        for dir in glob(ds_path+"/*"):
            if dir.endswith("cats"):
                label = 0
            elif dir.endswith("dogs"):
                label = 1
            else:
                warnings.warn("Directory {} must contain 'cat' or 'dog'")
            for f in glob(dir+"/*"+filetype):
                data.append(f)
                labels.append(label)
                
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = io.read_image(self.data[idx])
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return label, image

    def __len__(self):
        return len(self.data)