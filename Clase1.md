# Clase 1: Fundamentos y Anatomía de la Inteligencia Artificial

¡Bienvenidos a la cátedra! Tienen frente a ustedes una de las herramientas más potentes del mundo para el desarrollo de IA: laptops con arquitectura **NVIDIA Blackwell (RTX 5070 Ti)**. En este curso no solo aprenderemos teoría; vamos a ensuciarnos las manos con código y hardware real.

## 0. Preparación del Entorno

Antes de empezar, debemos asegurarnos de que estamos trabajando en el lugar correcto. En su terminal (Anaconda Prompt), observen el texto que aparece al principio de la línea entre paréntesis:

* **Si ven `(ia_fcyt)`:** Ya están dentro del entorno. No necesitan ejecutar nada más.
* **Si ven `(base)` o nada:** Deben ejecutar: `conda activate ia_fcyt`.

> **¿Por qué?** El entorno `ia_fcyt` contiene las librerías específicas (PyTorch, CUDA 12.8) que permiten que su código "hable" directamente con los núcleos de la GPU.

Luego, inicien su entorno de trabajo:

* Escriban `jupyter notebook` para abrir la interfaz en el navegador.
* O abran **VS Code** y asegúrense de que el "Kernel" (esquina superior derecha) diga `ia_fcyt`.

---

## 1. El Cambio de Paradigma: Programación vs. Aprendizaje

En las materias de programación que cursaron hasta ahora, ustedes escribían las reglas. En Inteligencia Artificial, el flujo cambia radicalmente:

* **Programación Tradicional:** $Datos + Reglas \rightarrow Salida$. (Ejemplo: Un `if` que decide si un número es par).
* **Machine Learning (ML):** $Datos + Salidas \rightarrow Reglas$. (Ejemplo: Le damos 10.000 fotos de piezas defectuosas y la IA deduce la regla para detectarlas).

### El "Músculo" tras la lógica (RTX 5070 Ti)

Aprender reglas a partir de datos es, en el fondo, un problema de **optimización matemática masiva**. Sus núcleos **Tensor** están diseñados para realizar billones de multiplicaciones de matrices por segundo. Lo que a una CPU le tomaría horas, a esta arquitectura le toma segundos.

---

## 2. Anatomía de un Dataset (El Combustible)

Un **Dataset** es una matriz de información. Para que la IA aprenda, los datos deben estar estructurados:

* **Muestras (Samples):** Son las **filas**. Cada una es un evento único (una lectura de sensor, un registro).
* **Características (Features - $X$):** Son las **columnas de entrada**. Lo que observamos (voltaje, frecuencia, dimensiones).
* **Etiqueta (Target/Label - $y$):** Es la **salida** que queremos predecir (¿Es falla?, ¿Es normal?, ¿Qué categoría es?).

> **Ing. Electrónicos:** Piensen en $X$ como señales de entrada y $y$ como el estado del sistema.
> **Ing. Informáticos:** Piensen en $X$ como atributos de un objeto y $y$ como su clase en una DB.

---

## 3. Práctica Inicial: Exploración de Datos

Abran un nuevo archivo `.ipynb` y ejecuten este bloque. Vamos a usar el dataset **Iris**, el "Hola Mundo" del ML, para entender la geometría de los datos.

```python
import pandas as pd
from sklearn.datasets import load_iris

# 1. Cargar datos
raw_data = load_iris()
df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
df['especie'] = raw_data.target

# 2. Visualizar la "forma" de los datos
print(f"Dimensiones del dataset: {df.shape}") # (Muestras, Características)
print("\nPrimeras 5 muestras (X) y su etiqueta (y):")
print(df.head())

```

### ¿Qué es un Tensor?

Verán que trabajamos con tablas de **Pandas**, pero la GPU no entiende de tablas. Para procesar esto en la RTX 5070 Ti, convertiremos estos datos en **Tensores**. Un tensor es simplemente una matriz multidimensional optimizada para vivir en la memoria de la placa de video (VRAM).

---

## 4. Rigor de Ingeniería: ¿Cómo usamos los datos?

**Nunca** usamos todos los datos para entrenar. Sería como darle a un alumno el examen resuelto para que "estudie": solo memorizaría las respuestas.

1. **Training Set (Entrenamiento):** El 80% de los datos. La IA los usa para ajustar sus pesos.
2. **Test Set (Prueba):** El 20% restante. Es el **examen final**. Son datos que la IA jamás ha visto.

**Regla de Oro:** Si tu modelo tiene 99% de precisión en entrenamiento pero 50% en prueba, tienes **Overfitting** (sobreajuste). Tu IA memorizó, no aprendió a generalizar.

---

## 5. Visualización: Ver antes de Entrenar

Un buen ingeniero siempre visualiza los datos antes de lanzar un modelo. Ejecuten esto para ver cómo se agrupan las especies de plantas según sus dimensiones:

```python
import matplotlib.pyplot as plt

# Graficamos: Largo del pétalo vs Ancho del pétalo
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df.iloc[:, 2], df.iloc[:, 3], c=df['especie'], cmap='viridis', edgecolors='k')
plt.xlabel(raw_data.feature_names[2])
plt.ylabel(raw_data.feature_names[3])
plt.title("Visualización de Clústeres de Datos (Iris Dataset)")
plt.colorbar(scatter, label="Especie (0, 1, 2)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

```

### Desafío para la clase:

Observen el gráfico generado. Hay grupos que se separan fácilmente y otros que se solapan.
**¿Podrían trazar una línea imaginaria que separe los grupos?** El algoritmo de IA que programaremos hará exactamente eso, pero en espacios de muchas más dimensiones.

---

**¿Lograron generar el gráfico?** Si es así, guarden sus avances. En la siguiente sección del repo veremos cómo realizar la **partición técnica de datos** (`train_test_split`) de forma profesional. ¿Quieren que avancemos con el código para separar sus datos de entrenamiento y prueba?
