import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

# Sabit değerler
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 3
IMAGE_SIZE = 224


# Veri yükleme sınıfı
class CityImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (string): CSV dosyasının yolu
            image_dir (string): Görüntülerin bulunduğu dizin
            transform (callable, optional): Görüntü üzerinde uygulanacak dönüşümler
        """
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # Şehir etiketlerini sayısal değerlere dönüştür
        self.city_to_label = {
            'Istanbul': 0,
            'Ankara': 1,
            'Izmir': 2
        }

        if 'city' in self.data_frame.columns:
            self.data_frame['label'] = self.data_frame['city'].map(self.city_to_label)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data_frame['filename'].iloc[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Eğitim veya test datasına göre farklı davran
        if 'label' in self.data_frame.columns:
            label = self.data_frame['label'].iloc[idx]
            return image, label
        else:
            return image, self.data_frame['filename'].iloc[idx]


# Veri ön işleme ve dönüşümleri
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    }


# ResNet modeli
class ResNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetModel, self).__init__()
        # Pretrained ResNet50 modelini yükle
        self.model = models.resnet50(pretrained=True)

        # Son fully connected katmanı değiştir
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Eğitim fonksiyonu
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # En iyi modeli kaydet
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    return model


# Ana eğitim ve tahmin fonksiyonu
def main():
    # Dönüşümleri tanımla
    transforms_dict = get_transforms()

    # Veri setini yükle
    train_dataset = CityImageDataset(
        csv_file=r"C:\Users\Kerem\Desktop\train_data.csv",
        image_dir=r"C:\Users\Kerem\Desktop\train\train",
        transform=transforms_dict['train']
    )

    # Train ve validation setlerini ayır
    train_subset, val_subset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    # DataLoader'ları oluştur
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Test setini yükle
    test_dataset = CityImageDataset(
        csv_file=r"C:\Users\Kerem\Desktop\test.csv",
        image_dir=r"C:\Users\Kerem\Desktop\test\test",
        transform=transforms_dict['test']
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Modeli oluştur
    model = ResNetModel(num_classes=NUM_CLASSES)

    # Loss ve optimizatör
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Modeli eğit
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS)

    # Test seti için tahminleri hazırla
    trained_model.load_state_dict(torch.load('best_model.pth'))
    trained_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)

    # Label'ları tersine çevir
    label_to_city = {v: k for k, v in train_dataset.city_to_label.items()}

    # Tahminleri kaydet
    predictions = []
    filenames = []
    with torch.no_grad():
        for images, batch_filenames in test_loader:
            images = images.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            batch_predictions = [label_to_city[pred.item()] for pred in predicted]
            predictions.extend(batch_predictions)
            filenames.extend(batch_filenames)

    # Sonuçları CSV olarak kaydet
    submission_df = pd.DataFrame({'filename': filenames, 'city': predictions})
    submission_df.to_csv('submission.csv', index=False)
    print("Tahminler tamamlandı ve submission.csv oluşturuldu!")


if __name__ == "__main__":
    main()
