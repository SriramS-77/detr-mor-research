import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define transforms
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the training data
voc_dataset = datasets.VOCDetection(
    root='./data', 
    year='2007', 
    image_set='train', 
    download=True, 
    transform=transform
)

print(f"VOC 2007 train images downloaded: {len(voc_dataset)}")

# Download and load the training data
voc_dataset = datasets.VOCDetection(
    root='./data', 
    year='2007', 
    image_set='test', 
    download=True, 
    transform=transform
)

print(f"VOC 2007 test images downloaded: {len(voc_dataset)}")

# Download and load the training data
voc_dataset = datasets.VOCDetection(
    root='./data', 
    year='2012', 
    image_set='train', 
    download=True, 
    transform=transform
)

print(f"VOC 2012 train images downloaded: {len(voc_dataset)}")
