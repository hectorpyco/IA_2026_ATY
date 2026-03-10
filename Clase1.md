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
Aquí tienes la continuación para tu archivo `.md`. He integrado las explicaciones técnicas sobre el dataset, la lógica de los colores en la visualización y el procedimiento profesional para la partición de datos.

---

## 6. Entendiendo el Dataset Iris y la "Magia" de los Colores

¿Por qué usamos estas flores en una clase de ingeniería? Porque representan el problema fundamental de la **Clasificación**.

### El Dataset: Tres Especies, Cuatro Dimensiones

El dataset contiene 150 muestras de tres especies: **Setosa** (0), **Versicolor** (1) y **Virginica** (2). Para cada una, medimos cuatro características (largo y ancho de sépalo y pétalo).

### ¿Cómo generamos los colores en el gráfico anterior?

En la línea `scatter = plt.scatter(..., c=df['especie'], cmap='viridis')`, ocurren dos procesos clave:

1. **`c=df['especie']`**: Le asignamos a cada punto el valor numérico de su etiqueta (0, 1 o 2).
2. **`cmap='viridis'`**: Es el **Colormap**. Es una matriz de colores que mapea números a una escala visual. El 0 se vuelve violeta, el 1 verde y el 2 amarillo.

**Interpretación de Ingeniería:**
Si observan su gráfico, verán que los puntos violetas (**Setosa**) están aislados. Decimos que son **linealmente separables**. Sin embargo, los verdes y amarillos se solapan. Aquí es donde la IA demuestra su valor: encontrando el límite óptimo donde el ojo humano duda.

---

## 7. Preparación para el Entrenamiento: Partición de Datos

Como ingenieros, no podemos confiar en un modelo que "memoriza". Debemos garantizar que la IA pueda **generalizar** ante datos que nunca ha visto antes. Para ello, realizamos la **Partición Técnica**.

Utilizaremos la regla del **80/20**:

* **$X$ (Características):** Las dimensiones de la planta.
* **$y$ (Etiqueta):** La especie.

Ejecuten el siguiente bloque para preparar sus conjuntos de datos:

```python
from sklearn.model_selection import train_test_split

# 1. Definimos nuestras variables de entrada (X) y salida (y)
X = df.drop('especie', axis=1) # Todas las columnas menos la etiqueta
y = df['especie']              # Solo la columna objetivo

# 2. Realizamos la división técnica
# test_size=0.2 separa el 20% para el examen final
# random_state=42 asegura que todos obtengamos los mismos datos (reproducibilidad)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Verificamos la distribución
print(f"Muestras totales: {len(df)}")
print(f"Muestras para Entrenamiento (Estudio): {len(X_train)}")
print(f"Muestras para Test (Examen): {len(X_test)}")

```

### ¿Por qué `random_state=42`?

En ingeniería, la **reproducibilidad** es ley. Si no fijamos una "semilla" aleatoria, cada vez que ejecuten el código los datos se mezclarán de forma distinta, y sus resultados no serán comparables. Usamos 42 por convención en la comunidad de IA, pero podría ser cualquier número.

---

## 8. El Concepto de "Garbage In, Garbage Out" (GIGO)

Antes de pasar al entrenamiento en la siguiente sesión, recuerden este principio: **Si entra basura, sale basura.** La potencia de sus **RTX 5070 Ti** no sirve de nada si el dataset está mal etiquetado o tiene ruidos excesivos. La IA no es mágica; es un espejo de la calidad de sus datos.

---

**¿Ya tienen sus datos divididos y listos en la memoria de la laptop?** Si es así, guarden este Notebook. En la próxima sección vamos a definir nuestro primer **Modelo de Clasificación** y veremos cómo la GPU calcula la frontera de decisión automáticamente. ¿Les gustaría que prepare el código para entrenar su primer algoritmo de Machine Learning?
