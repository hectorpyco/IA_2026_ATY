# IA_2026_ATY
Para proceder de manera segura y evitar errores de ejecución más adelante, realizaremos una **verificación técnica por niveles**. Esto asegurará que el hardware (RTX 5070) esté correctamente vinculado al software.

Abre una **Terminal** (PowerShell) en la laptop y sigue estos pasos:

---

### Paso 1: Verificación del Gestor de Entornos (Conda)

Primero debemos confirmar que Miniconda está instalado y accesible desde la terminal.

**Ejecuta este comando:**

```powershell
conda --version

```

* **Resultado esperado:** Debería devolver algo como `conda 23.x.x` o superior.
* **Si falla:** Significa que Miniconda no se instaló o no se agregó al PATH de Windows.

---

### Paso 2: Verificación del Entorno Específico

Debemos asegurarnos de que el entorno que creamos para la materia está activo.

**Ejecuta estos comandos:**

```powershell
conda activate ia_fcyt
python --version

```

* **Resultado esperado:** La terminal debería mostrar `(ia_fcyt)` al principio de la línea y la versión de Python debería ser `3.10.x`.

---

### Paso 3: Verificación de Librerías y Aceleración por GPU

Este es el paso más importante. Vamos a verificar que Python puede "hablar" con la placa de video **NVIDIA RTX 5070**.

**Copia y pega este comando único en la terminal (es un mini-script de Python):**

```powershell
python -c "import torch; print('\n' + '='*30); print(f'PyTorch versión: {torch.__version__}'); print(f'¿GPU disponible?: {torch.cuda.is_available()}'); print(f'Nombre GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DETECTADA'}'); print('='*30)"

```

**Analicemos el resultado:**

1. **¿GPU disponible?: True** $\rightarrow$ ¡Perfecto! Estamos listos para Deep Learning.
2. **¿GPU disponible?: False** $\rightarrow$ **Atención:** PyTorch se instaló en modo CPU. Debes reinstalarlo con el comando de soporte CUDA que vimos antes.

---

### Paso 4: Verificación de Librerías de Datos

Confirmemos que las herramientas de nivelación (NumPy, Pandas, Scikit-Learn) están presentes.

**Ejecuta:**

```powershell
python -c "import numpy as np; import pandas as pd; import sklearn; print('Librerías de datos OK')"

```

---

### Paso 5: Verificación del IDE (VS Code)

1. Abre **Visual Studio Code**.
2. Presiona `Ctrl + Shift + P` y escribe: **"Python: Select Interpreter"**.
3. Asegúrate de seleccionar el que dice **'ia_fcyt': conda**.
4. Crea un archivo nuevo llamado `test.ipynb` (un Jupyter Notebook).
5. En la esquina superior derecha del editor, verifica que el **Kernel** seleccionado sea el entorno `ia_fcyt`.

---

### Resumen de Estado

| Componente | Comando de prueba | Estado esperado |
| --- | --- | --- |
| **Conda** | `conda --version` | Versión visible |
| **Entorno** | `conda activate ia_fcyt` | Prefijo (ia_fcyt) activo |
| **CUDA / GPU** | Script del Paso 3 | True (RTX 5070) |
| **Jupyter** | Abrir `.ipynb` | Kernel conectado |

---

**¿Algún paso te devolvió un error o el resultado del Paso 3 fue "False"?** Si todo está en "True", estamos listos para cargar el primer dataset y empezar con la práctica de Machine Learning. ¿Te gustaría que te proporcione el código para cargar un dataset de prueba?
