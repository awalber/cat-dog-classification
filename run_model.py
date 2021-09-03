import os
from glob import glob
import warnings
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.io as io
import torchvision.models as models
import torchvision.transforms as T

class MyDataset(Dataset):
    def __init__(self, ds_path, filetype=".jpg", transform=None):
        super(MyDataset,self).__init__()
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

def train(model,train_dl,optimizer,criterion,num_epoch,weights,device,train_ratio=0.8):
    if os.path.isfile(weights):
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint)
    num_train = int(len(train_dl)*train_ratio)
    num_valid = len(train_dl) - num_train
    training, validation = random_split(train_dl, [num_train,num_valid])
    datasets = {"Training":training.dataset, "Validation":validation.dataset}
    best_val_loss = np.inf
    no_improvement = 0
    for epoch in range(num_epoch):
        for d in datasets:
            if d == "Training":
                model.train()
            else:
                model.eval()
            dataset = datasets[d]
            total_pts = 0
            running_loss, running_acc = 0.0, 0.0
            for i, sample in enumerate(dataset):
                labels, data = sample
                labels, data = labels.to(device), data.to(device)
                optimizer.zero_grad()
                out = model(data)
                _, pred = torch.max(out, 1)
                num_correct = (pred == labels).sum()
                loss = criterion(out,labels)
                if d == "Training":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                running_acc  += num_correct.data.item()
                total_pts += len(sample[0])

            print("Epoch {}, {} Loss: {}, Accuracy: {}".format(epoch + 1, d, running_loss / i, running_acc / total_pts * 100))
            if d == "Validation":
                val_loss = running_loss / i
                if np.array(val_loss) < best_val_loss:
                    torch.save(model.state_dict(),weights)
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement == 3:
                    break
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint)
    return model

def test(model,test_dl,device,weights=None):
    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint)
    model.eval()
    total_pts = 0
    running_acc = 0.0
    for sample in test_dl:
        labels, data = sample
        labels, data = labels.to(device), data.to(device)
        out = model(data)
        _, pred = torch.max(out, 1)
        num_correct = (pred == labels).sum()
        running_acc += num_correct.data.item()
        total_pts += len(sample[0])
    print("Test Data: Accuracy: {}".format(running_acc / total_pts * 100))

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    transform = {
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
    data_dir = r"E:\Data\cat_dog"
    weights_path = r"E:\Data\cat_dog\weights.pt"
    num_epoch = 10
    batch_size = 512
    num_workers= 7
    pretrained = True
    num_classes = 2
    train_data = os.path.join(data_dir,"training_set")
    test_data = os.path.join(data_dir,"test_set")
    train_ds = MyDataset(train_data,transform=transform["train"])
    test_ds = MyDataset(test_data,transform=transform["test"])
    model = models.resnet18(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, num_classes)
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters())
    train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    model = train(model,train_dl,optimizer,criterion,num_epoch,weights_path,device)
    test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    test(model,test_dl,device,weights=weights_path)