import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from run_model import Classifier
import torch.optim as optim
import torch.nn as nn

batch_size = 256
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam
parent_dir = os.path.dirname(os.path.dirname(__file__))
weights = os.path.join(parent_dir,"model_weights","weights.pt")

class Data(BaseModel):
    name : str
    path : str
    description : Optional[str] = None
    label : Optional[str] = None



model = Classifier(batch_size,optimizer,criterion,weights=weights)
app = FastAPI()

@app.post("/predict")
def create_item(data : Data):
    print(data.path)
    print(data)
    path = os.path.abspath(data.path)
    label,prob = model.predict(path)
    return {"prediction":label,"probability":prob}