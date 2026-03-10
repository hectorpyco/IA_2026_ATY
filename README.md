Este es el manual técnico paso a paso para configurar las **14 laptops Acer Predator** con arquitectura **Blackwell (RTX 5070 Ti)**. Ejecuta estos comandos en el **Anaconda Prompt**.

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
pip install numpy pandas matplotlib scikit-learn jupyter notebook

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

## Procedimiento de Inicio de Clase

Para que los alumnos empiecen a trabajar, los comandos diarios son:

1. `conda activate ia_fcyt`
2. `jupyter notebook` (o abrir VS Code y seleccionar el kernel `ia_fcyt`).

---

