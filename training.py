from torchvision.transforms.v2 import Compose, Resize, RandomHorizontalFlip, ToImage, ToDtype
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch



## Image transforms
transforms = Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True),
    Resize((500,600)), # zmienia rozmiar obrazkana 500x600 (format wysokość x szerokość)
    RandomHorizontalFlip() # z obraca obrazek poziom z prawdopodobieństwem 0.5
])

training_dataset = ImageFolder(root="train", transform=transforms)
print("Dataset loaded")

print(training_dataset[0])
