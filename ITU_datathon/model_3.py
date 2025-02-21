import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import os
import torch.nn as nn

# Veri yolları
train_data_path = r"C:\Users\Kerem\Desktop\train_data.csv"
test_data_path = r"C:\Users\Kerem\Desktop\test.csv"
train_images_path = r"C:\Users\Kerem\Desktop\train\train"
test_images_path = r"C:\Users\Kerem\Desktop\test\test"
output_path = r"C:\Users\Kerem\Desktop\test_results.csv"

# Şehirleri etiketlere dönüştürme
sehirler = {
    "Istanbul": 0,
    "Ankara": 1,
    "Izmir": 2,
}

# Verileri yükleme
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

train_df["city"] = train_df["city"].replace(sehirler)

class CustomDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None, test_mode=False):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.test_mode = test_mode

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test_mode:
            return image, self.data_frame.iloc[idx, 0]  # filename
        else:
            label = int(self.data_frame.iloc[idx, 1])
            return image, label

transform = transforms.Compose([
    transforms.RandomRotation(15),  # Rastgele döndürme
    transforms.RandomHorizontalFlip(),  # Yatay yansıma
    transforms.RandomVerticalFlip(),  # Dikey yansıma
    transforms.RandomResizedCrop(224),  # Rastgele zoom (yakınlaştırma)
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Parlaklık ve kontrast       #hepsi sonradan eklendi
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Kaydırma
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = CustomDataset(
    data_frame=train_df,
    root_dir=train_images_path,
    transform=transform
)

test_dataset = CustomDataset(
    data_frame=test_df,
    root_dir=test_images_path,
    transform=transform,
    test_mode=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # İlk konvolüsyonel blok
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Daha fazla filtre
        self.pool1 = nn.MaxPool2d(2, 2)

        # İkinci konvolüsyonel blok
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Üçüncü konvolüsyonel blok
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Dördüncü konvolüsyonel blok
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fully connected katmanlar
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)  # Daha büyük bağlantı
        self.fc2 = nn.Linear(1024, 3)  # 3 sınıf

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = x.view(-1, 512 * 14 * 14)  # Flatten
        x = F.relu(self.fc1(x))  # Fully connected 1
        x = self.fc2(x)  # Fully connected 2

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")
model = CNN_Model().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005) #AdamW yaptık ve lr yi yarıya indirdik

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Öğrenme oranını güncelleme
        scheduler.step()                     #buda yeni eklendi

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')


def test_model(model, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(zip(filenames, predicted.cpu().numpy()))

    result_df = pd.DataFrame(predictions, columns=['filename', 'city'])
    result_df['city'] = result_df['city'].replace({v: k for k, v in sehirler.items()})
    result_df.to_csv(output_path, index=False)
    print(f"Tahminler {output_path} dosyasına kaydedildi.")

# Modeli eğit ve test et
train_model(model, train_loader, criterion, optimizer, scheduler , num_epochs=50) #epoch sayısı 1000 yapıldı ve scheduler de ekledik
test_model(model, test_loader)


# 3.denemede de amaç , farklı model kullanmak bakalım

#50 epoch la deniycez