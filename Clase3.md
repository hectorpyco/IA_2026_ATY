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

  * **Repositorio en Kaggle:** [NEU Surface Defect Database]([https://www.google.com/search?q=https://www.kaggle.com/datasets/kaustubhb93/neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)) (Fuente común para acceso rápido).
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

¿Te parece si ahora pasamos a definir la **Arquitectura CNN Clase 3**, diseñada para extraer el máximo provecho de los núcleos Tensor de sus laptops?
