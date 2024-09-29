import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

import os
from torch.utils.data import Dataset
from PIL import Image

class MRIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all image paths in the directory (including subdirectories)
        self.image_paths = [os.path.join(root, fname) 
                            for root, _, fnames in os.walk(data_dir) 
                            for fname in fnames if fname.endswith('.jpg')]

        print(f"Found {len(self.image_paths)} images in {data_dir}")
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Extract the label based on the folder name in the path
        # Example path: 'C:/Users/golla/OneDrive/Desktop/NeuroNet_new/neuronet/data/train/Moderate Dementia/OAS1_0308_MR1_mpr-1_101.jpg'
        # Split the path to get the disorder folder name
        label_folder = os.path.basename(os.path.dirname(img_path))  # Get the folder name containing the image

        # Define the mapping from folder name to numerical label
        label_mapping = {
            
            "Mild Dementia": 0,
            "Moderate Dementia": 1,
            "Non Demented": 2,
            "very mild Dementia" :3
        }
        
        # Get the label based on the folder name
        label = label_mapping.get(label_folder, -1)  # Return -1 if label is not found
        
        if self.transform:
            image = self.transform(image)

        return image, label


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 62 * 62, 128)
        self.fc2 = nn.Linear(128, 4)  # Adjust based on the number of classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 62 * 62)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(train_dir, val_dir, epochs=10, batch_size=16, learning_rate=0.001):
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")

    # Check if the train directory exists
    if not os.path.exists(train_dir):
        print(f"Training directory does not exist: {train_dir}")
        return

    # Check if the validation directory exists
    if not os.path.exists(val_dir):
        print(f"Validation directory does not exist: {val_dir}")
        return

    # List files in training directory
    print("Files in training directory:")
    print(os.listdir(train_dir))

    # List files in validation directory
    print("Files in validation directory:")
    print(os.listdir(val_dir))
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_dataset = MRIDataset(train_dir, transform=transform)
    val_dataset = MRIDataset(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%')

    torch.save(model.state_dict(), 'model/trained_model.pt')


if __name__ == '__main__':
    train_model('C:\\Users\\golla\\OneDrive\\Desktop\\NeuroNet_new\\neuronet\\data\\train', 
                'C:\\Users\\golla\\OneDrive\\Desktop\\NeuroNet_new\\neuronet\\data\\val')
