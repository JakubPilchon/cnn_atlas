from torchvision.transforms.v2 import Compose, Resize, RandomHorizontalFlip, ToImage, ToDtype
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import json

#from model_architecture import Model
from model_architecture import Model

# musze obczaić o co biega z gpu, bo muszę władować dataloader do GPU ale ciężej jest z tym niż myślałem
#device = ("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Model will be trained on {device}")

## Image transforms
transforms = Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True), # zmienia typ danych na float32 i normalizuje nam dane
    Resize((500,600)), # zmienia rozmiar obrazkana 500x600 (format wysokość x szerokość)
    RandomHorizontalFlip() # z obraca obrazek poziom z prawdopodobieństwem 0.5
])

# pytorch oferuje fajną klase specjalnie do naszego typu zbiorów danych
training_dataset = ImageFolder(root="train", transform=transforms)
test_dataset = ImageFolder(root="valid", transform=transforms)


# Dataloader od Datasetu różni się tym że 1) Dataloader jest z batchowany 2) Dataloader jest generatorem, więc pozwala nam nieco oszczędzić na pamięci
training_dataloader = DataLoader(training_dataset,
                                 batch_size=32,
                                 shuffle= True)

test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle= True)

#training_dataloader.to(device)
#test_dataloader.to(device)
print("Dataset loaded")


model = Model("test_dropout.json")


# uczenie modelu
model.fit(training_dataloader,
         test_dataloader,
         model_dir="models",
         epochs=23)

