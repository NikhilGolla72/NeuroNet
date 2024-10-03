import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        label_folder = os.path.basename(os.path.dirname(img_path))  # Get the folder name containing the image

        # Define the mapping from folder name to numerical label (case-sensitive)
        label_mapping = {
            "Mild Dementia": 0,
            "Moderate Dementia": 1,
            "Non Demented": 2,
            "very mild Dementia": 3  # Corrected case sensitivity
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
        self.fc1 = nn.Linear(32 * 14 * 14, 128)  # Adjusted size for 64x64 images
        self.fc2 = nn.Linear(128, 4)  # 4 classes: No Dementia, Mild, Moderate, Very Mild

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 14 * 14)  # Adjusted size for 64x64 images
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(train_dir, val_dir, epochs=10, batch_size=32, learning_rate=0.001):
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")

    # Check if the train and validation directories exist
    if not os.path.exists(train_dir):
        print(f"Training directory does not exist: {train_dir}")
        return

    if not os.path.exists(val_dir):
        print(f"Validation directory does not exist: {val_dir}")
        return

    # Transform: Resize images to a smaller size (64x64) for faster training
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Smaller size
        transforms.ToTensor(),
    ])
    
    train_dataset = MRIDataset(train_dir, transform=transform)
    val_dataset = MRIDataset(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # Added workers
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)    # Added workers

    model = SimpleNN().to(device)  # Move model to GPU if available
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scaler = torch.amp.GradScaler()  # Updated mixed precision training

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):  # Updated mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}')

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%')

    torch.save(model.state_dict(), 'trained_model.pt')
    print('Model training complete and saved as trained_model.pt')


if __name__ == '__main__':
    train_model('C:\\Users\\manda\\OneDrive\\Desktop\\Neuronet\\NeuroNet\\data\\train', 
                'C:\\Users\\manda\\OneDrive\\Desktop\\Neuronet\\NeuroNet\\data\\val')
