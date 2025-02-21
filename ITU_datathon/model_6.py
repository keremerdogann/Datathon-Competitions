# Tek dosyada birleştirilmiş sokak sınıflandırma modeli kodu

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json

# --- DATASET.PY ---
class StreetDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.train = train
        self.image_paths = []
        self.labels = []

        # Şehir etiketlerini topla
        cities = sorted(os.listdir(data_dir))
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(cities)

        # Görüntüy yollarını ve etiketleri topla
        for city in cities:
            city_path = os.path.join(data_dir, city)
            if os.path.isdir(city_path):
                for img_name in os.listdir(city_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(city_path, img_name))
                        self.labels.append(city)

        self.labels = self.label_encoder.transform(self.labels)

        # Varsayılan dönüşümler
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def get_num_classes(self):
        return len(self.label_encoder.classes_)

    def get_class_names(self):
        return self.label_encoder.classes_

# --- MODEL.PY ---
class StreetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(StreetClassifier, self).__init__()
        self.base_model = models.resnet50(weights='IMAGENET1K_V2')
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# --- TRAIN.PY ---
def train_model(data_dir, num_epochs=50, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = StreetDataset(data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = dataset.get_num_classes()
    model = StreetClassifier(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix({'loss': f'{loss.item():.4f}',
                                   'acc': f'{100.*train_correct/train_total:.2f}%'})

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_bar.set_postfix({'loss': f'{loss.item():.4f}',
                                     'acc': f'{100.*val_correct/val_total:.2f}%'})

        val_losses.append(val_loss / len(val_loader))
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss}, 'best_model.pth')

        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_curves.png')

# --- PREDICT.PY ---
def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StreetClassifier(num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_image(image_path, model, label_encoder):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    device = next(model.parameters()).device
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_idx].item()

    predicted_city = label_encoder.inverse_transform([predicted_idx])[0]
    return predicted_city, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Street Classification Pipeline')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Predict the city from an image')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to the dataset')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the model file')
    parser.add_argument('--image_path', type=str, help='Path to the image file for prediction')
    args = parser.parse_args()

    if args.train:
        train_model(data_dir=args.data_dir)
    elif args.predict:
        if not args.image_path:
            print("Error: --image_path argument is required for prediction.")
        else:
            dataset = StreetDataset(data_dir=args.data_dir)
            label_encoder = dataset.label_encoder
            num_classes = dataset.get_num_classes()
            model = load_model(args.model_path, num_classes)
            city, confidence = predict_image(args.image_path, model, label_encoder)
            print(f'Predicted City: {city}, Confidence: {confidence:.2f}')
    else:
        print("Error: Please specify either --train or --predict.")

