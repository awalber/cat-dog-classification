import os
from typing import Dict
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.io as io
import torchvision.models as models
import torchvision.transforms as T

from data import ClassifierDataset

class Classifier:
    def __init__(self, batch_size, optimizer, criterion, num_classes=2, weights=None, device="cuda") -> None:
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        self.model.to(device)
        self.batch_size = batch_size
        self.optimizer = optimizer(params=self.model.parameters()) # set the model parameters to the optimizer
        self.criterion = criterion
        self._weights = weights
        self.device = device
        self.transform = {
            "train":T.Compose([
                T.ToPILImage(),
                T.Resize((256,256)),
                T.RandomCrop(224),
                T.RandomRotation(90),
                T.RandomVerticalFlip(0.25),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "test":T.Compose([
                T.ToPILImage(),
                T.Resize((256,256)),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }

    def load_weights(self):
        checkpoint = torch.load(self.weights)
        self.model.load_state_dict(checkpoint)

    @property
    def weights(self):
        if self._weights is not None:
            pass
        else:
            f = os.path.abspath(__file__)
            parent_dir = os.path.dirname(os.path.dirname(f))
            os.makedirs(os.path.join(parent_dir,"model_weights"),exist_ok=True)
            self._weights = os.path.join(parent_dir,"model_weights","weights.pt")

        return self._weights

    def run_one_epoch(self, datasets : Dict):
        for d in datasets:
            if d == "Training":
                self.model.train()
            else:
                self.model.eval()
            dataset = datasets[d]
            total_pts = 0
            running_loss, running_acc = 0.0, 0.0
            for sample in dataset:
                labels, data = sample
                labels, data = labels.to(self.device), data.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(data)
                _, pred = torch.max(out, 1)
                num_correct = (pred == labels).sum()
                loss = self.criterion(out,labels)
                if d == "Training":
                    loss.backward()
                    self.optimizer.step()
                running_loss += loss.item()
                running_acc  += num_correct.data.item()
                total_pts += len(sample[0])
        return running_loss, running_acc, total_pts

    def train(self, train_data, num_epoch, train_ratio = 0.8, num_workers = 1):
        train_ds = ClassifierDataset(train_data,transform=self.transform["train"])
        train_dl = DataLoader(train_ds,batch_size=self.batch_size,shuffle=True,num_workers=num_workers)
        if os.path.isfile(self.weights):
            self.load_weights()
        else:
            open(self.weights,"w")
        num_train = int(len(train_dl)*train_ratio)
        num_valid = len(train_dl) - num_train
        training, validation = random_split(train_dl, [num_train,num_valid])
        datasets = {"Training":training.dataset, "Validation":validation.dataset}
        best_val_loss = np.inf
        no_improvement = 0
        for epoch in range(num_epoch):
            for name, dataset in datasets.items():
                running_loss, running_acc, total_pts = self.run_one_epoch({name:dataset})
                print("Epoch {}, {} Loss: {}, Accuracy: {}".format(epoch + 1, name, running_loss / total_pts, running_acc / total_pts * 100))
                if name == "Validation":
                    val_loss = running_loss / total_pts
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(),self.weights)
                        no_improvement = 0
                    else:
                        no_improvement += 1
                    if no_improvement == 3:
                        break
    
    def test(self, test_data, num_workers = 1):
        test_ds = ClassifierDataset(test_data,transform=self.transform["test"])
        test_dl = DataLoader(test_ds,batch_size=self.batch_size,shuffle=False,num_workers=num_workers)
        if os.path.isfile(self.weights):
            self.load_weights()
        self.model.eval()
        total_pts = 0
        running_acc = 0.0
        for sample in test_dl:
            labels, data = sample
            labels, data = labels.to(self.device), data.to(self.device)
            out = self.model(data)
            _, pred = torch.max(out, 1)
            num_correct = (pred == labels).sum()
            running_acc += num_correct.data.item()
            total_pts += len(sample[0])
        print("Test Data: Accuracy: {}".format(running_acc / total_pts * 100))

    def predict(self, data : str):
        # predicts a single value instead of a directory of values
        if os.path.isfile(self.weights):
            self.load_weights()
        self.model.eval()
        image = io.read_image(data)
        image = self.transform["test"](image).unsqueeze(0)
        image = image.to(self.device)
        out = self.model(image)
        prob = nn.Softmax(dim=-1)(out)
        _, pred = torch.max(out, 1)
        if pred.item() == 0:
            label = "cat"
        elif pred.item() == 1:
            label = "dog"
        prob = prob[0,pred].item()
        return label, prob


if __name__ == "__main__":
    print(torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam
    data_dir = r"E:\Data\cat_dog"
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_path = os.path.join(parent_dir,"model_weights","weights.pt")
    num_epoch = 10
    batch_size = 64
    train_data = os.path.join(data_dir,"training_set")
    test_data = os.path.join(data_dir,"test_set")
    c = Classifier(batch_size,optimizer,criterion,weights=weight_path)
    label,prob = c.predict(r"C:\Users\awalb\Downloads\test.jpg")
    print("Image is predicted to be a {} with {:.1%} probability").format(label,prob)
    
