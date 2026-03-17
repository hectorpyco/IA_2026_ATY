Este es el manual técnico paso a paso para configurar las **14 laptops Acer Predator** con arquitectura **Blackwell (RTX 5070 Ti)**. Ejecuta una sola vez estos comandos en el **Anaconda Prompt**.

---

## Manual de Configuración: Laboratorio IA (FCyT UNCA)

### 1. Creación y Preparación del Entorno

Crea un espacio aislado para evitar conflictos con otras materias y activa el entorno.

```bash
# Crear el entorno con Python 3.10
conda create -n ia_fcyt python=3.10 -y

# Activar el entorno (obligatorio antes de instalar nada)
conda activate ia_fcyt

```

### 2. Instalación del "Cerebro" (PyTorch + CUDA 12.8)

Para la serie 50 (RTX 5070 Ti), es crítico usar la versión de **CUDA 12.8** para evitar el error de incompatibilidad de arquitectura (`sm_120`).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

```

### 3. Instalación de Herramientas de Datos e Interfaz

Instala el stack necesario para el análisis de señales y procesamiento de datos.

```bash
pip install numpy pandas matplotlib scikit-learn jupyter notebook seaborn

```

### 4. Verificación de Hardware y Software (Script de Prueba)

Una vez terminadas las instalaciones, ejecuta este comando único para validar que la GPU esté lista para trabajar.

```bash
python -c "import torch, numpy, pandas, sklearn; print('\n' + '='*45 + '\n[REPORTE DE SISTEMA - FCyT UNCA]\n' + f'GPU Detectada: {torch.cuda.get_device_name(0)}\nCUDA Disponible: {torch.cuda.is_available()}\nCapacidad de Cómputo: {torch.cuda.get_device_capability(0)}\nLibrerías de Datos: OK\n' + '='*45)"

```

**Resultado esperado:**

* **GPU Detectada:** NVIDIA GeForce RTX 5070 Ti Laptop GPU.
* **CUDA Disponible:** True.
* **Capacidad de Cómputo:** (12, 0).

---

# Guía Técnica: Comprendiendo el Poder del Hardware (Compute Capability 12.0)

¡Bienvenidos a la cátedra de Inteligencia Artificial! Para empezar, han notado que al ejecutar el script de verificación, sus laptops reportaron una **Capacidad de Cómputo (Compute Capability) de 12.0**. ¿Qué significa esto realmente para un ingeniero?

### 1. ¿Qué es la Compute Capability?

No es una medida de velocidad de reloj (MHz), sino la **versión de la arquitectura del hardware**. Define qué instrucciones técnicas puede ejecutar la GPU a nivel físico.

El número **12.0** nos indica que estamos trabajando sobre la arquitectura **Blackwell** de NVIDIA (la más avanzada al 2026). Para que se den una idea de la evolución:

| Arquitectura | Compute Capability | Año Dominante | Hito Tecnológico |
| --- | --- | --- | --- |
| **Pascal** | 6.x | 2016 | Inicios del Deep Learning masivo. |
| **Ampere** | 8.x | 2020 | Introducción de los Tensor Cores de 3ra Gen. |
| **Ada Lovelace** | 9.0 | 2023 | Salto en eficiencia y Ray Tracing. |
| **Blackwell** | **12.0** | **2025/26** | **Optimización para Transformers y FP8.** |

---

### 2. ¿Por qué es importante el "12.0" para esta materia?

Tener una capacidad de 12.0 significa que sus laptops no solo son rápidas, sino que tienen unidades de hardware dedicadas que otras computadoras no poseen:

* **Tensor Cores de 5ta Generación:** Son núcleos diseñados exclusivamente para la multiplicación de matrices. Mientras que una CPU procesa números uno a uno, estos núcleos procesan **tensores** (bloques de datos) en un solo ciclo de reloj.
* **Soporte Nativo FP8:** En IA, no siempre necesitamos precisión decimal infinita. La arquitectura Blackwell permite trabajar con precisión de 8 bits sin perder exactitud, lo que duplica la velocidad de entrenamiento y reduce el consumo de memoria.
* **Transformer Engine:** Incluye hardware especializado para acelerar los "Transformers" (la arquitectura detrás de ChatGPT y la visión artificial moderna).

---

### 3. Del Bit al Tensor: La Matemática en la GPU

En esta materia, dejaremos de pensar en variables aisladas $x, y$ para pensar en **Tensores** $\mathcal{T}$. Matemáticamente, una operación de IA se ve así:

$$\mathbf{Y} = \sigma(\mathbf{W} \cdot \mathbf{X} + \mathbf{b})$$

Donde $\mathbf{W}$ representa los "pesos" o el conocimiento de la IA. Gracias a la potencia **12.0**, sus GPUs pueden calcular estas ecuaciones sobre millones de datos en milisegundos.

---

### 4. Flujo de Trabajo en Clase (Workflow)

Para sacar provecho a este hardware, seguiremos siempre este orden en nuestros archivos `.ipynb`:

1. **Ingeniería de Datos:** Limpiar y normalizar el Dataset usando la CPU (Core Ultra 9).
2. **Carga en VRAM:** Enviar los datos a la memoria de la GPU RTX 5070 Ti.
3. **Entrenamiento/Inferencia:** Ejecutar los cálculos pesados en los núcleos CUDA/Tensor.
4. **Recuperación:** Traer los resultados de vuelta para visualizarlos.

> **Nota para el alumno:** Una laptop potente no reemplaza un mal algoritmo. Si sus datos son mediocres (*Garbage In*), el resultado será mediocre (*Garbage Out*), solo que lo obtendrán mucho más rápido.

---

### Ejercicio de Reflexión para el Repo:

*Investiguen y respondan en su primer archivo de notas:*
¿Cuál es la diferencia técnica entre un **Núcleo CUDA** (propósito general) y un **Núcleo Tensor** (especializado en IA)? ¿Por qué la arquitectura Blackwell (12.0) pone tanto énfasis en estos últimos?

---

