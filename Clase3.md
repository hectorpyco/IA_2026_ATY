Para esta **Clase 3**, el objetivo es que el alumno entienda que, aunque cambiamos de "flores" a "láminas de acero", la estructura del código en PyTorch es prácticamente la misma. Esto refuerza la confianza del estudiante al ver que los conceptos son repetibles.

---

# Clase 3: Visión Artificial e Inspección Industrial

En la clase anterior usamos un "cañón para matar una mosca" con el dataset Iris. Hoy, finalmente le daremos a nuestra arquitectura **NVIDIA Blackwell (RTX 5070 Ti)** un desafío a su altura. Pasaremos de datos tabulares (números en tablas) a **datos no estructurados**: píxeles organizados en tensores.

## 1\. El Salto a la Visión: ¿Qué es una imagen para una IA?

Para el ojo humano, una imagen es una representación continua de formas y colores. Para un modelo de **Deep Learning**, una imagen es un **Tensor Multidimensional**.

  * **Escala de grises:** Se representa como una matriz de 2 dimensiones ($H \times W$). Cada celda contiene un valor de intensidad luminosa, típicamente entre $0$ (negro) y $255$ (blanco).
  * **Color (RGB):** Se representa como un tensor de 3 dimensiones ($C \times H \times W$), donde $C$ son los canales de color (Rojo, Verde, Azul).
  * **En PyTorch:** El estándar para procesar imágenes es el formato `[Batch, Channels, Height, Width]`.

-----

## 2\. El Dataset: NEU Surface Defect Database

Para esta cátedra utilizaremos el **NEU Surface Defect Database**, un estándar internacional en la investigación de control de calidad industrial.

### Origen y Contexto Académico

Este dataset fue desarrollado por el **Laboratorio de Automatización de la Universidad del Noreste (NEU)** en China. Su objetivo es proporcionar una base sólida para el entrenamiento de algoritmos de inspección automática en la industria del acero.

  * **Publicación original:** *Bao, Y., et al. (2014). "A surface defect detection algorithm for hot-rolled steel strip based on classification of defect categories."*
  * **Contenido:** 1,800 imágenes en formato bitmap (.bmp).
  * **Resolución:** Cada imagen es de $200 \times 200$ píxeles.
  * **Clases:** 300 muestras por cada uno de los 6 tipos de defectos.

### Clasificación de Defectos Industriales

| Clase | Nombre Técnico | Descripción Visual |
| :--- | :--- | :--- |
| **Cr** | *Crazing* (Agrietamiento) | Red de grietas finas en la superficie. |
| **In** | *Inclusion* (Inclusión) | Materia extraña atrapada durante el laminado. |
| **Pa** | *Patches* (Parches) | Áreas de material irregular o superpuesto. |
| **PS** | *Pitted Surface* (Superficie picada) | Pequeños orificios o cavidades. |
| **RS** | *Rolled-in Scale* (Escama laminada) | Residuos de óxido comprimidos en el metal. |
| **Sc** | *Scratches* (Arañazos) | Líneas longitudinales por fricción mecánica. |

-----

## 3\. Recursos y Documentación

Para profundizar en la naturaleza de estos datos y cómo han sido utilizados en el estado del arte (SOTA), pueden consultar los siguientes enlaces:

  * **Repositorio en Kaggle:** [NEU Surface Defect Database]([(https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)] (Fuente común para acceso rápido).
  * **Paper de referencia:** [vía IEEE Xplore]([https://www.google.com/search?q=https://www.researchgate.net/publication/261350170_A_surface_defect_detection_algorithm_for_hot-rolled_steel_strip](https://ieeexplore.ieee.org/document/11267867)) para entender la metodología original de captura.

-----

## 4\. Rigor de Ingeniería: Precisión vs. Seguridad

En la industria, la precisión no es solo un porcentaje. Un **Falso Negativo** (una grieta no detectada) puede resultar en la falla estructural de una pieza y poner en riesgo vidas humanas. Por ello, en esta clase aprenderemos que el modelo no solo debe ser "inteligente", sino **robusto** ante variaciones de iluminación y ruido en la captura, aprovechando la capacidad de cómputo de sus estaciones de trabajo para realizar entrenamientos más profundos.

-----

**¿Lograron identificar los 6 tipos de defectos en las imágenes de muestra?** Si es así, estamos listos para pasar a la siguiente sección: **Data Augmentation y Carga de Tensores**, donde prepararemos las imágenes para que entren en la memoria de la **RTX 5070 Ti**.

---

## 5. Refuerzo de Conceptos: La Tubería de Datos (Data Pipeline)

Sin importar el problema, un ingeniero de IA siempre repite estos 4 pasos. Hoy los profundizaremos para el procesamiento de imágenes:

1.  **Transformación:** Adecuar la imagen (redimensionar a $224 \times 224$ y normalizar tensores).
2.  **Carga (Loading):** Mapear las subcarpetas del dataset NEU a objetos de Python.
3.  **Loteado (Batching):** Agrupar imágenes en bloques (batches) para procesamiento paralelo.
4.  **Transferencia:** Mover el lote de la memoria RAM a la **VRAM (cuda:0)** de la RTX 5070 Ti.

---


## 5.1. Adquisición y Organización de Datos (Paso Crítico)

Ejecuten este bloque **antes** del código de carga. Este script creará las carpetas, descargará el dataset (aprox. 70MB) y lo dejará listo para la **RTX 5070 Ti**.

```python
import os
import zipfile
import urllib.request

# 1. Definir rutas
data_dir = './data'
dataset_path = os.path.join(data_dir, 'NEU')
zip_name = "NEU_Surface_Defect.zip"
# URL directa a un espejo del dataset NEU
url = "https://github.com/Yue-Gao/NEU-Surface-Defect-Database/archive/refs/heads/master.zip"

# 2. Crear carpetas si no existen
os.makedirs(data_dir, exist_ok=True)

# 3. Descarga del dataset
if not os.path.exists(zip_name):
    print("Descargando dataset NEU... esto puede tardar un momento.")
    urllib.request.urlretrieve(url, zip_name)
    print("Descarga completada.")

# 4. Extracción
if not os.path.exists(dataset_path):
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    # Renombrar la carpeta extraída para que coincida con nuestro código
    os.rename(os.path.join(data_dir, 'NEU-Surface-Defect-Database-master'), dataset_path)
    print(f"Dataset extraído en: {dataset_path}")
else:
    print("El dataset ya existe en el disco. Procediendo...")
```

---

## 6. Código: Preparando los Tensores de Imagen (Explicado)

Ahora que los archivos existen en `./data/NEU`, el código para que las etiquetas sean legibles:

```python
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# 1. Transformaciones: El "molde" para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

# 2. Carga: PyTorch "mapea" las carpetas
# ImageFolder entrará a 'data/NEU' y verá las subcarpetas como 'Crazing', 'Inclusion', etc.
dataset = datasets.ImageFolder(root='./data/NEU', transform=transform)

# 3. Partición Técnica: 80% para estudiar, 20% para el examen
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size], 
                                     generator=torch.Generator().manual_seed(42))

# 4. DataLoader: La tubería hacia la GPU
# Aquí es donde el Batch de 32 se prepara para entrar a los núcleos Tensor
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Estructura lista. Clases: {dataset.classes}")
print(f"Imágenes para entrenamiento: {len(train_data)}")
```



---

### ¿Por qué esta estructura de carpetas?

`ImageFolder` es una función de conveniencia de PyTorch. Para que funcione, la estructura en tu disco debe verse así:

```text
data/NEU/
├── Crazing/     --> (Clase 0) imagen1.bmp, imagen2.bmp...
├── Inclusion/   --> (Clase 1) imagen1.bmp, imagen2.bmp...
├── Patches/     --> (Clase 2) ...
└── ...
```

Si tus alumnos ven el mensaje **"Dataset cargado. Total de imágenes: 1800"**, significa que ya tenemos los datos en la "línea de salida".

---

**¿Lograron descargar y extraer el dataset en las laptops?** Si la respuesta es **Sí**, ya no hay nada que nos detenga. 


## 7. Concepto Repetido: La Red Neuronal Convolucional (CNN)

Para imágenes, ya no usamos el Perceptrón simple de la clase anterior. Usamos una **CNN**.
* **¿Por qué?** Porque las capas convolucionales actúan como filtros espaciales que detectan bordes, texturas y patrones geométricos complejos.

### Anatomía de la CNN que construiremos:
1.  **Convolución:** Extrae rasgos (como la forma de una grieta).
2.  **Pooling:** Reduce la resolución espacial, haciendo al modelo robusto ante traslaciones.
3.  **Totalmente Conectada:** Capas densas finales que clasifican el rasgo detectado en uno de los 6 defectos.


[Image of convolutional neural network layers]](https://www.plainconcepts.com/wp-content/uploads/2024/11/Convolutional-Neural-Network-developers-neural-network.webp)


---

## 8. El Hardware: Monitoreo de VRAM

Mientras corra el entrenamiento (en el siguiente paso), quiero que abran el **Administrador de Tareas** (Rendimiento -> GPU).
* Observen el incremento en la **Memoria de video dedicada**.
* Noten la actividad en los **Núcleos Tensor** (Tensor Cores).

**Pregunta técnica para la clase:**
Si una imagen comprimida en disco pesa $50\text{ KB}$, ¿por qué el uso de VRAM sube a **Gigabytes**?
*(Pista: No solo cargamos píxeles; almacenamos mapas de características, gradientes de error y estados del optimizador en alta precisión de 32 bits).*

---

## 9. Práctica: Visualización del "Lote" de Datos

Antes de entrenar, verifiquen la integridad del cargador de datos. Ejecuten esto para visualizar una grilla de las láminas de acero:

```python
import matplotlib.pyplot as plt
import torchvision
import numpy as np

def mostrar_lote(loader):
    dataiter = iter(loader)
    imagenes, etiquetas = next(dataiter)
    
    # Des-normalizar para visualización
    imagenes = imagenes / 2 + 0.5 
    
    grid = torchvision.utils.make_grid(imagenes)
    plt.figure(figsize=(12, 8))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap='gray')
    plt.title("Lote de Inspección Industrial (NEU Dataset)")
    plt.axis('off')
    plt.show()

mostrar_lote(train_loader)
```

---

**¿Lograron visualizar la grilla de láminas de acero en sus notebooks?** Si la visualización es correcta, el "pipeline" de datos está listo.

Excelente, Héctor. El documento tiene una estructura de ingeniería impecable. Ahora que los alumnos ya "ven" los datos, el siguiente paso lógico es construir el "cerebro" que los procesará.

Aquí tienes los puntos **10** (Definición de la arquitectura) y **11** (El ciclo de entrenamiento), listos para que los agregues a tu archivo `.md`.

---

## 10. Construcción del "Cerebro": Arquitectura CNN Industrial

Vamos a traducir el diagrama de la **CNN** a código real. Usaremos dos bloques convolucionales seguidos de una etapa de clasificación. Noten cómo cada capa tiene una función específica en la detección de fallas.

```python
import torch.nn as nn
import torch.nn.functional as F

class CNN_Industrial(nn.Module):
    def __init__(self):
        super(CNN_Industrial, self).__init__()
        
        # Bloque 1: Detector de bordes y texturas simples
        # Entrada: 1 canal (gris), Salida: 16 mapas de rasgos
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # Reduce el tamaño a la mitad (112x112)
        
        # Bloque 2: Detector de patrones complejos (grietas, manchas)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Después del 2do pool, la imagen queda en 56x56
        
        # Etapa de Clasificación (Capas Densas)
        # 32 mapas de 56x56 píxeles = 100,352 entradas
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 6) # 6 neuronas de salida (una por defecto)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Aplanamos el tensor para entrar a las capas densas
        x = x.view(-1, 32 * 56 * 56)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instanciamos y movemos a la RTX 5070 Ti (cuda:0)
modelo = CNN_Industrial().to("cuda")
print(modelo)
```

---

## 11. Entrenamiento de Alto Rendimiento: Sacando jugo a la VRAM

Aquí es donde el hardware de sus laptops marca la diferencia. Ejecutaremos el ciclo de entrenamiento enviando cada lote de imágenes directamente a los **Núcleos Tensor**.

**Instrucción:** Mientras ejecutan este bloque, mantengan abierto el Administrador de Tareas para observar el pico de consumo de energía y memoria de la GPU.

```python
import torch.optim as optim

# 1. Definimos la "Regla de Evaluación" (Loss) y el "Optimizador"
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

# 2. Bucle de Entrenamiento
epochs = 5
print(f"Iniciando entrenamiento en: {torch.cuda.get_device_name(0)}")

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # TRANSFERENCIA: Movemos imágenes y etiquetas a la GPU
        inputs, labels = data[0].to("cuda"), data[1].to("cuda")

        # Limpiar gradientes
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = modelo(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Época {epoch + 1} - Error promedio: {running_loss / len(train_loader):.4f}")

print("Entrenamiento finalizado.")
```

---

## 12. Análisis de Resultados: ¿Es confiable el sistema?

Una vez que el modelo "estudió", debemos evaluar su precisión con el **Test Set**. No olviden que en ingeniería industrial, un error puede ser catastrófico.

```python
correct = 0
total = 0
modelo.eval() # Modo evaluación (apaga capas de entrenamiento)

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to("cuda"), data[1].to("cuda")
        outputs = modelo(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Precisión en las {total} imágenes de prueba: {100 * correct / total:.2f}%')
```

---

**¿Qué porcentaje de precisión obtuvieron?** Dada la potencia de la RTX 5070 Ti, el entrenamiento debería ser sumamente rápido. 

---

## 13. Auditoría de Ingeniería: Matriz de Confusión

Un 90% o 95% de precisión suena bien, pero en una línea de producción, **no todos los errores valen lo mismo**. Confundir un rasguño superficial (*Scratch*) con una grieta estructural (*Crazing*) puede detener una fábrica innecesariamente o, peor aún, dejar pasar una pieza peligrosa.

Ejecuten este bloque para ver dónde "duda" su modelo:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

modelo.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = modelo(images.to("cuda"))
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Generar y graficar la matriz
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel('Predicción de la IA')
plt.ylabel('Defecto Real')
plt.title('Matriz de Confusión Industrial')
plt.show()
```

---

## 14. Reflexión Final: El Rol del Ingeniero en la Era de la IA

Hoy han visto cómo la arquitectura **NVIDIA Blackwell** procesa miles de imágenes en segundos. Sin embargo, la potencia sin control no sirve de nada. 

Como ingenieros de la **FCyT**, su trabajo no es solo escribir `modelo.train()`. Su valor reside en:
1.  **Entender los Datos:** Saber que si la iluminación en la fábrica cambia, el modelo puede fallar.
2.  **Interpretar el Error:** Saber que un Falso Negativo en una grieta es un riesgo inaceptable.
3.  **Desplegar Soluciones:** Una IA que solo vive en un Jupyter Notebook no le sirve a la industria. Necesitamos interfaces que el operario pueda usar.



---

## 15. Tarea: Del Notebook a la Aplicación Real

El desafío para la próxima clase es convertir este modelo en una herramienta funcional. 

**Consigna:** Desarrollar un script o pequeña interfaz (pueden usar librerías como `Gradio`, `Tkinter` o simplemente una función de carga de archivos en Python) que permita:
1.  **Cargar una imagen externa** (una que no esté en el dataset original).
2.  **Preprocesarla** (llevarla a 224x224 y escala de grises).
3.  **Pasarla por el modelo** entrenado hoy.
4.  **Mostrar en pantalla la predicción:** "El sistema detecta: [Tipo de Defecto] con un X% de confianza".

**Pista técnica:** Para predecir una sola imagen, recuerden que PyTorch espera un lote. Deberán usar `imagen.unsqueeze(0)` para convertir su imagen de $[1, 224, 224]$ a un "lote de uno" $[1, 1, 224, 224]$.

---

**¿Todos tienen su precisión final anotada?** Guarden su modelo con `torch.save(modelo.state_dict(), 'modelo_neu.pth')` para poder usarlo en su tarea sin tener que re-entrenar.

