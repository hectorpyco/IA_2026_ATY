

# Clase 6: Visión Artificial Industrial - Dataset NEU

En esta sesión, utilizaremos la potencia de nuestra **RTX 5070 Ti** para procesar datos no estructurados: píxeles organizados en tensores para detectar fallas en láminas de acero.

---

## 1. Adquisición y Organización de Datos (Paso Crítico)
Ejecuten este bloque primero. Este script descargará el dataset desde el repositorio de la cátedra, lo extraerá y normalizará las rutas para que el resto del cuaderno funcione sin errores.

```python
import os
import zipfile
import urllib.request
import shutil

# 1. Configuración de rutas
data_dir = './data'
dataset_path = os.path.join(data_dir, 'NEU')
url = "https://github.com/hectorpyco/IA_2026_ATY/archive/refs/heads/main.zip"
zip_name = "IA_CAT_DATA.zip"

# 2. Descarga del repositorio
if not os.path.exists(zip_name):
    print("Descargando dataset desde el repositorio de la cátedra...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(zip_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print("Descarga completada.")

# 3. Extracción y Reorganización para PyTorch
if not os.path.exists(dataset_path):
    print("Extrayendo y normalizando estructura...")
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Ruta interna del ZIP generado por GitHub
    source_folder = os.path.join(data_dir, 'IA_2026_ATY-main', 'NEU_Data', 'NEU-DET')
    
    if os.path.exists(source_folder):
        shutil.move(source_folder, dataset_path)
        print(f"✅ Dataset listo en: {dataset_path}")
        shutil.rmtree(os.path.join(data_dir, 'IA_2026_ATY-main'))
    else:
        print("❌ Error: No se encontró la carpeta NEU-DET en la ruta esperada.")
```

---

## 2. Preparación de Tensores e Imágenes
Cargamos las librerías de PyTorch y definimos las transformaciones. Redimensionaremos a $224 \times 224$ y convertiremos a escala de grises para optimizar el entrenamiento.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

# Configuración de rutas locales
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'validation')

# Transformaciones estándar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

# Carga de carpetas (ImageFolder)
train_set = datasets.ImageFolder(root=train_path, transform=transform)
val_set = datasets.ImageFolder(root=val_path, transform=transform)

# DataLoaders (Tubería hacia la GPU)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(val_set, batch_size=32, shuffle=False)

print(f"✅ Clases detectadas: {train_set.classes}")
print(f"Imágenes cargadas: {len(train_set)} (Train) / {len(val_set)} (Val)")
```

---

## 3. Arquitectura de la CNN Industrial
Definimos el "cerebro" del modelo. Usaremos capas convolucionales para extraer texturas de las láminas de acero.

```python
class NEU_Classifier(nn.Module):
    def __init__(self):
        super(NEU_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 224 -> Pool -> 112 -> Pool -> 56
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 6) # 6 tipos de defectos

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Aplanado (Flatten)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Enviar modelo a la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NEU_Classifier().to(device)
print(f"🚀 Ejecutando en: {torch.cuda.get_device_name(0)}")
```

---

## 4. Ciclo de Entrenamiento
Ejecuten este bloque para iniciar el aprendizaje. La **RTX 5070 Ti** debería completar las 10 épocas en pocos minutos.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
print("Iniciando entrenamiento...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Época [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")
```

---

## 5. Inferencia (Prueba de Campo)
Para cerrar, seleccionen cualquier imagen del set de validación para verificar la predicción del modelo.

```python
import matplotlib.pyplot as plt

def predict_image(image_path, model, transform):
    img = Image.open(image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        clase = train_set.classes[pred.item()]
        
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicción del Sistema: {clase}")
    plt.axis('off')
    plt.show()

# Prueba con una imagen real
predict_image('./data/NEU/validation/inclusion/inclusion_275.jpg', model, transform)
```

---

### Tarea de Ingeniería
Al finalizar la clase, guarden su modelo entrenado en su carpeta de Documentos:
`torch.save(model.state_dict(), 'modelo_neu_clase6.pth')`

---
