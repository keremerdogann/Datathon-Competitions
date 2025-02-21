import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy.typing as npt

train_data_path = r"C:\Users\Kerem\Desktop\train_data.csv"
test_data_path = r"C:\Users\Kerem\Desktop\test.csv"

sehirler = {
    "Istanbul":0,
    "Ankara":1,
    "Izmir":2,
}

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)


train_df["city"] = train_df["city"].replace(sehirler)

# Verilerin ilk 5 satırını yazdıralım
print("Train Data:")
print(train_df.head())
print("\nTest Data:")
print(test_df.head())


class CustomDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame  # DataFrame
        self.root_dir = root_dir  # Görsellerin bulunduğu ana dizin (train/test)
        self.transform = transform  # Görsel dönüşümleri

    def __len__(self):
        return len(self.data_frame)  # Veri setindeki örnek sayısı

    def __getitem__(self, idx):
        # Görsel yolunu oluşturma
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")  # Görseli aç ve RGB'ye dönüştür
        label = int(self.data_frame.iloc[idx, 1])  # Etiketi al ve int'e çevir

        if self.transform:
            image = self.transform(image)  # Dönüşümü uygula

        return image, label

# Görüntü ön işleme ve veri artırma işlemleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Görüntü boyutlandırma
    transforms.ToTensor(),  # Görüntüyü tensöre dönüştürme
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizasyon
])

# Train ve Test veri setlerinin oluşturulması
train_dataset = CustomDataset(
    data_frame=train_df,
    root_dir=r"C:\Users\Kerem\Desktop\train\train",  # Alt klasöre kadar tam yol
    transform=transform
)
test_dataset = CustomDataset(
    data_frame=test_df,
    root_dir=r"C:\Users\Kerem\Desktop\test\test",  # Alt klasöre kadar tam yol
    transform=transform
)

# DataLoader tanımlama
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print("Veriler başarıyla yüklendi!")


import torch.nn as nn
import torch.optim as optim

# CNN modelini tanımlama
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # İlk konvolüsyonel blok
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3x224x224 -> 32x224x224
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x224x224 -> 32x112x112

        # İkinci konvolüsyonel blok
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32x112x112 -> 64x112x112
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x112x112 -> 64x56x56

        # Üçüncü konvolüsyonel blok
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64x56x56 -> 128x56x56
        self.pool3 = nn.MaxPool2d(2, 2)  # 128x56x56 -> 128x28x28

        # Dördüncü konvolüsyonel blok
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128x28x28 -> 256x28x28
        self.pool4 = nn.MaxPool2d(2, 2)  # 256x28x28 -> 256x14x14

        # Fully connected katmanlar
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # 256x14x14 -> 512
        self.fc2 = nn.Linear(512, 3)  # 512 -> 3 (şehir sınıfları)

    def forward(self, x):
        # Konvolüsyonel blok 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Konvolüsyonel blok 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Konvolüsyonel blok 3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Konvolüsyonel blok 4
        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        # Tam bağlı katmanlara geçiş
        x = x.view(-1, 256 * 14 * 14)  # Flattening
        x = F.relu(self.fc1(x))  # Fully connected 1
        x = self.fc2(x)  # Fully connected 2 (çıktı)

        return x

# Modeli oluşturma
model = CNN_Model()

# Modeli GPU'ya taşıma (varsa)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if device == "cuda":
    print("Cudayı kullanıyor cihaz.")


# Kayıp fonksiyonu ve optimizasyonu tanımlama
criterion = nn.CrossEntropyLoss()  # Kayıp fonksiyonu
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

# Modeli eğitme fonksiyonu
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Modeli eğitim moduna al
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Sıfırlama
            optimizer.zero_grad()

            # İleri geçiş
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Geriye doğru geçiş
            loss.backward()
            optimizer.step()

            # İlerleme
            running_loss += loss.item()

            # Doğru tahmin sayısını hesapla
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%')

# Modeli eğitme
train_model(model, train_loader, criterion, optimizer, num_epochs=10)


# Test fonksiyonu
def test_model(model, test_loader):
    model.eval()  # Modeli test moduna al
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Modeli test etme
test_model(model, test_loader)

# Yeni bir görüntü üzerinde tahmin yapma
def predict_city(image_path):
    img = Image.open(image_path)
    img = transform(img)  # Ön işleme
    img = img.unsqueeze(0).to(device)  # Batch boyutunu ekle ve GPU'ya taşı

    # Model ile tahmin
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    # Şehir isimlerini dönüştür
    city_names = {0: 'Istanbul', 1: 'Ankara', 2: 'Izmir'}
    predicted_city = city_names[predicted.item()]
    return predicted_city

# Örnek tahmin
print(predict_city(r"C:\Users\Kerem\Desktop\datathn\test\image_1000.jpg"))
