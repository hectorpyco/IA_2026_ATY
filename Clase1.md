Para empezar "despacito", vamos a establecer los cimientos del **Machine Learning (ML)**. No empezaremos con código complejo, sino con la **anatomía del dato**.

---

## 1. El Cambio de Paradigma

En la informática tradicional, tú programas las reglas. En ML, tú entregas los datos y el hardware (la GPU) deduce las reglas.

* **Programación Tradicional:** $Datos + Reglas = Salida$
* **Machine Learning:** $Datos + Salidas = Reglas$

### El Rol del Hardware (RTX 5070 Ti)

¿Por qué necesitamos esta potencia para "aprender"? Porque el aprendizaje es, matemáticamente, una **optimización masiva**. El modelo realizará miles de millones de multiplicaciones de matrices para encontrar la "regla" que mejor se ajusta a tus datos. Tus núcleos **Tensor** están diseñados específicamente para eso.

---

## 2. Anatomía de un Dataset

Un **Dataset** es una matriz de información. Para que la IA aprenda, debemos estructurarla correctamente.

### Conceptos Clave:

* **Muestras (Samples):** Son las filas. Cada fila es un evento único (una lectura de sensor, un registro de alumno).
* **Características (Features - $X$):** Son las columnas de entrada. Lo que observamos (voltaje, temperatura, largo de un pétalo).
* **Etiqueta (Target/Label - $y$):** Es lo que queremos predecir (¿El motor falló?, ¿Es una estafa?, ¿Qué tipo de planta es?).

> **Nota para Electrónica:** Piensen en $X$ como las señales de entrada de un sistema y en $y$ como el estado de la salida.
> **Nota para Informática:** Piensen en $X$ como los atributos de un objeto y en $y$ como su categoría.

---

## 3. Práctica Inicial: Carga y Exploración (Jupyter)

Pide a los alumnos que abran un **Jupyter Notebook** y ejecuten este bloque. Vamos a usar el dataset **Iris**, que aunque es clásico, es perfecto para entender la geometría de los datos.

```python
import pandas as pd
from sklearn.datasets import load_iris

# 1. Cargar datos
raw_data = load_iris()
df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
df['especie'] = raw_data.target

# 2. Visualizar la "forma" de los datos
print(f"Dimensiones del dataset: {df.shape}") # (Filas, Columnas)
print("\nPrimeras 5 muestras:")
print(df.head())

```

### El Concepto de "Tensor"

Para que la GPU procese esto, eventualmente convertiremos estas tablas en **Tensores**. Un tensor es simplemente una matriz generalizada que puede vivir en la memoria de la placa de video.

---

## 4. ¿Cómo se usa un Dataset adecuadamente?

Aquí es donde aplicamos rigor de ingeniería. **Nunca** usamos todos los datos para entrenar a la IA.

1. **Conjunto de Entrenamiento (Training):** Los datos con los que la IA estudia.
2. **Conjunto de Prueba (Testing):** El examen final con datos que la IA nunca vio.

**Regla de Oro:** Si la IA saca 100/100 en el entrenamiento pero falla en la prueba, no ha aprendido, solo ha memorizado (**Overfitting**).

---

## 5. Ejercicio de Nivelación: Visualización de Características

Antes de que la GPU trabaje, el ingeniero debe "ver" los datos. Pídeles que ejecuten esto para ver cómo se relacionan las variables:

```python
import matplotlib.pyplot as plt

# Graficamos: Largo del pétalo vs Ancho del pétalo
plt.figure(figsize=(8, 5))
plt.scatter(df.iloc[:, 2], df.iloc[:, 3], c=df['especie'], cmap='viridis')
plt.xlabel(raw_data.feature_names[2])
plt.ylabel(raw_data.feature_names[3])
plt.title("Visualización de Clústeres de Datos")
plt.colorbar(label="Especie")
plt.show()

```

### Pregunta para la clase:

*Observando el gráfico:* ¿Creen que una máquina podría trazar una línea para separar los grupos de colores? **Esa línea es el modelo de IA que vamos a construir.**

---

**¿Todos los alumnos logran ver el gráfico de puntos de colores en sus laptops?** Si es así, estamos listos para realizar la división técnica de los datos y preparar el primer entrenamiento real. ¿Te gustaría que procedamos con el código para la **partición de datos**?
