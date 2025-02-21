import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F

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

# Veri dönüşümleri
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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

# VGG Modelini Özelleştirme
class VGG_Model(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG_Model, self).__init__()
        # Önceden eğitilmiş VGG16 modelini yükle
        self.vgg = models.vgg16(pretrained=True)

        # "features" kısmını dondur
        for param in self.vgg.features.parameters():
            param.requires_grad = False

        # VGG'nin son katmanını güncelle
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg(x)

# Cihaz seçimi ve model tanımlama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")

model = VGG_Model(num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Eğitim fonksiyonu
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=50):
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

        scheduler.step()  # Öğrenme oranını güncelle

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# Test fonksiyonu
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
train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=50)
test_model(model, test_loader)

#BURADA AĞIRLIKLARI DONDURARAK İŞLEM YAPIYORUZ, CHATGPT YE FARKINI SOR