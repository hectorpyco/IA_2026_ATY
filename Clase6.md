


# Clase 6: Visión Artificial e Inspección Industrial de Metales

En esta sesión, daremos el salto de los datos tabulares a los **datos no estructurados**. Utilizaremos la potencia de cálculo de nuestra **RTX 5070 Ti** para procesar píxeles organizados en tensores, con el objetivo de automatizar el control de calidad en la industria siderúrgica.

## 1. El Dataset: NEU Surface Defect Database

El dataset **NEU** (Northeastern University Surface Defect Database) es un referente mundial en la investigación de visión artificial aplicada a la metalurgia. 

### Origen y Propósito
Fue desarrollado para resolver uno de los problemas más críticos en la fabricación de acero: la detección de defectos superficiales en láminas laminadas en caliente. En una línea de producción real, el acero se desplaza a velocidades que hacen imposible la inspección humana fiable. Por ello, se entrenan redes neuronales para identificar fallas en tiempo real.



### Composición Técnica
El dataset se compone de **1,800 imágenes** de microscopía de luz (divididas en 1,440 para entrenamiento y 360 para validación) que cubren los 6 defectos más comunes:

1.  **Crazing (Agrietamiento):** Red de grietas finas que parecen "piel de cocodrilo".
2.  **Inclusion (Inclusión):** Impurezas no metálicas atrapadas en la superficie.
3.  **Patches (Parches):** Áreas con texturas o colores irregulares por oxidación o presión.
4.  **Pitted Surface (Picaduras):** Pequeños pozos o porosidad en el metal.
5.  **Rolled-in Scale (Cascarilla Laminada):** Óxidos que se incrustan en el acero durante el laminado.
6.  **Scratches (Rasguños):** Marcas lineales causadas por fricción mecánica.



---

## 2. El Salto a la Visión: ¿Qué es una imagen para una IA?

Para el ojo humano, una imagen es una representación de formas. Para un modelo de **Deep Learning**, una imagen es un **Tensor Multidimensional**.

* **Escala de Grises:** Se representa como una matriz de 2 dimensiones ($H \times W$). Cada celda contiene un valor de intensidad luminosa entre $0$ (negro) y $255$ (blanco).
* **En PyTorch:** El estándar para procesar estas imágenes es el formato `[Batch, Channels, Height, Width]`.
    * **Channels:** En nuestro caso es $1$ (Gris).
    * **Height/Width:** Redimensionamos a $224 \times 224$ para estandarizar la entrada a la red neuronal.



---

## 3. Arquitectura: La Red Neuronal Convolucional (CNN)

A diferencia de las redes simples (MLP), una CNN está diseñada para entender la **geometría espacial**.

1.  **Capas de Convolución:** Funcionan como "escáneres" que buscan patrones específicos (una línea recta indica un *Scratch*, una textura rugosa indica *Pitted Surface*).
2.  **Capas de Pooling:** Reducen el tamaño de la imagen, permitiendo que la red se enfoque solo en los rasgos más importantes y ahorrando memoria **VRAM** de nuestra GPU.
3.  **Capas Fully Connected:** Toman todas las características detectadas y deciden a cuál de las 6 categorías pertenece la imagen.

---

## 4. Laboratorio Práctico (Bloques de Código)

> **Instrucción para el alumno:** Copien los siguientes bloques en su Notebook de Jupyter uno por uno, asegurándose de ejecutar el paso de descarga antes de intentar cargar los datos.



## 4.1. Adquisición y Organización de Datos (Paso Crítico)
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
    source_folder = os.path.join(data_dir, 'IA_2026_ATY-main', 'NEU_Data')
    
    if os.path.exists(source_folder):
        shutil.move(source_folder, dataset_path)
        print(f"✅ Dataset listo en: {dataset_path}")
        shutil.rmtree(os.path.join(data_dir, 'IA_2026_ATY-main'))
    else:
        print("❌ Error: No se encontró la carpeta NEU-DET en la ruta esperada.")
```

---

## 4.2. Preparación de Tensores e Imágenes
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

## 4.3. Arquitectura de la CNN Industrial
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

## 4.4. Ciclo de Entrenamiento
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

## 4.5. Inferencia (Prueba de Campo)
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

# Tarea de Ingeniería: El Desafío del Inspector
Como ingenieros de la FCyT, su trabajo no termina en el Jupyter Notebook. El éxito de una IA se mide por su desempeño en el "mundo salvaje".

Paso A: Guardar el modelo
Al finalizar la clase, aseguren su trabajo guardando los pesos entrenados:
torch.save(model.state_dict(), 'modelo_neu_clase6.pth')

Paso B: El Desafío Externo (Validación Real)
Busquen en Google Imágenes o bases de datos metalúrgicas fotos de "Steel surface defects" que NO pertenezcan al dataset NEU.

Descarguen al menos 3 imágenes de diferentes defectos.

Guárdenlas en su carpeta de trabajo.

Usen la función predict_external() para ver si su modelo es capaz de identificar el defecto en una foto que jamás ha visto.

Pregunta Crítica para el informe:
¿El modelo mantiene una confianza alta (>80%) en imágenes de internet? ¿Qué factores (iluminación, zoom, resolución) creen que hacen que la IA se confunda?
