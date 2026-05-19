# Guía de Consolidación e Ingeniería en Visión por Computadora: Del Laboratorio a la Producción Industrial

Este material de resumen técnico consolida los conceptos fundamentales, las arquitecturas y los paradigmas de ingeniería aplicados a lo largo de los tres talleres prácticos de visión por computadora: **Inspección Industrial de Metales (NEU)**, **Clasificación de Moneda Nacional (PYG)** y **Detección y Rastreo de Tráfico en Tiempo Real (YOLO)**.

El diseño curricular de estos laboratorios representa una evolución lineal en la complejidad del procesamiento de imágenes: partiendo de la clasificación estática de datos altamente estructurados en entornos controlados, pasando por la ingeniería de datos manual y arquitecturas eficientes para dispositivos embebidos, hasta culminar en el procesamiento de flujos de video en tiempo real de misión crítica.

---

## Eje Temático 1: Fundamentos del Tensor y Clasificación Estácica (Caso: Inspección Industrial de Metales)

La base de la visión por computadora moderna descansa sobre el procesamiento de representaciones matriciales multidimensionales. El objetivo principal de este bloque fue comprender cómo una imagen se transforma en un flujo matemático procesable por una red neuronal convolucional (CNN).

### 1.1 La Estructura del Tensor 4D en PyTorch

Para el procesamiento eficiente en unidades de procesamiento gráfico (GPU), las imágenes individuales se consolidan en una estructura tensorial de cuatro dimensiones expresada como:

$$\text{Shape} = [\text{Batch Size}, \text{Channels}, \text{Height}, \text{Width}]$$

* **Batch Size (Lote):** Número de imágenes procesadas simultáneamente en un paso hacia adelante (*forward pass*). Controla la estabilidad del gradiente y el uso de la memoria VRAM.
* **Channels (Canales):** Profundidad del mapa de color. En imágenes en escala de grises (como el dataset NEU) el valor es $1$; en imágenes a color (RGB) el valor es $3$.
* **Height & Width (Alto y Ancho):** Dimensiones espaciales fijas de los píxeles. La red requiere que todas las entradas dentro del mismo *batch* posean dimensiones idénticas.

### 1.2 Mecánica de la Convolución y Extracción de Características

La operación fundamental de una CNN es la convolución discreta en dos dimensiones. Un filtro o *kernel* (una matriz pequeña de pesos entrenables, típicamente de $3\times3$ o $5\times5$) se desplaza a lo largo de las dimensiones de alto y ancho de la imagen.

* **Operación matemática:** Se calcula el producto punto elemento a elemento entre el *kernel* y la región local de la imagen, generando un mapa de características (*Feature Map*).
* **Jerarquía visual:** Las primeras capas convolucionales aprenden características primitivas de baja frecuencia (bordes, contrastes, esquinas). Las capas intermedias y profundas combinan estos mapas para detectar texturas complejas, patrones geométricos y semántica del objeto.

### 1.3 Estabilización y Regularización del Aprendizaje

Entrenar una red desde cero (*from scratch*) presenta desafíos de convergencia matemática:

* **Batch Normalization (BatchNorm):** Normaliza las activaciones de las capas ocultas para cada lote de entrenamiento. Esto mitiga el problema del *Internal Covariate Shift* (desplazamiento interno de covariables), permitiendo tasas de aprendizaje más altas y actuando como un regularizador débil.
* **Dropout:** Técnica de regularización donde se "apagan" u omiten aleatoriamente una fracción de neuronas (por ejemplo, el 30% o 50%) durante cada paso de entrenamiento. Esto evita la co-adaptación de características y fuerza a la red a aprender representaciones redundantes y robustas, combatiendo directamente el sobreajuste (*overfitting*).

### 1.4 Métricas de Evaluación en Control de Calidad

En entornos industriales, la métrica de *Accuracy* (Precisión global) suele ser engañosa debido al desbalance de clases (por ejemplo, la presencia de defectos es mucho más rara que el producto sano). Por ello, se aplican métricas basadas en la Matriz de Confusión:

* **Precision (Precisión):** Proporción de predicciones positivas que fueron correctas. Evita los falsos positivos.
* **Recall (Sensibilidad):** Proporción de positivos reales que fueron detectados correctamente. En la industria, un *Recall* bajo es crítico, ya que implica dejar pasar un defecto al cliente final.
* **F1-Score:** Media armónica entre *Precision* y *Recall*, ofreciendo un balance objetivo en datasets desbalanceados.

### 1.5 El Fenómeno del Domain Shift

El *Domain Shift* (desplazamiento de dominio) ocurre cuando la distribución estadística de los datos de entrenamiento (*source domain*) difiere de los datos de operación real (*target domain*). En la inspección industrial, un cambio en el ángulo de la cámara, el tipo de iluminación o el desgaste del rodillo de laminado puede hacer que un modelo con 99% de precisión en el laboratorio falle drásticamente en producción si no se entrena con variabilidad de datos.

---

## Eje Temático 2: El Paradigma Data-Centric y Eficiencia en el Borde (Caso: Clasificador de Billetes PYG)

El segundo bloque práctico desplazó el enfoque desde la arquitectura del modelo hacia la ingeniería de datos, introduciendo el concepto de aprendizaje por transferencia (*Transfer Learning*) enfocado en la computación en el borde (*Edge Computing*).

### 2.1 Enfoque Model-Centric vs. Data-Centric AI

* **Model-Centric:** Mantener los datos fijos e iterar constantemente en la modificación de la arquitectura de la red (añadir capas, cambiar funciones de activación) para mejorar el rendimiento.
* **Data-Centric (Andrew Ng):** Mantener la arquitectura fija y enfocar los esfuerzos de ingeniería en mejorar sistemáticamente la calidad, consistencia y diversidad del dataset. En el clasificador de billetes, la robustez no provino de usar una red más compleja, sino de mejorar la captura de datos y el etiquetado.

### 2.2 Mecánica del Transfer Learning (Aprendizaje por Transferencia)

Diseñar y entrenar una red profunda requiere millones de imágenes y semanas de cómputo. El *Transfer Learning* resuelve esto reutilizando un modelo previamente entrenado en un dataset masivo (como ImageNet, que contiene 1.4 millones de imágenes distribuidas en 1,000 clases).

* **Feature Extraction (Extractor de Características):** Se congelan los pesos de las capas convolucionales (*backbone*) fijando el parámetro `requires_grad = False`. El modelo retiene su capacidad general para reconocer formas, colores y texturas.
* **Rediseño del Classifier Head:** Se remueve la capa de salida original y se acopla una nueva estructura de capas densas (`nn.Linear`) adaptada al número de clases objetivo (las denominaciones de la moneda nacional). Solo estos nuevos pesos son optimizados durante el entrenamiento.

### 2.3 Arquitecturas Eficientes: MobileNetV2

Para desplegar modelos en terminales de autoservicio o dispositivos con recursos de hardware limitados, las CNN tradicionales resultan demasiado pesadas. **MobileNetV2** solventa este límite mediante el uso de **Convoluciones Separables en Profundidad (Depthwise Separable Convolutions)**:

* Divide la convolución estándar en dos pasos: una convolución por canal (*depthwise*) seguida de una convolución lineal $1\times1$ (*pointwise*).
* **Impacto matemático:** Reduce drásticamente el número de parámetros y las operaciones de punto flotante (FLOPs), permitiendo inferencias en milisegundos sin sacrificar significativamente la precisión.

### 2.4 Data Augmentation Avanzada para Entornos Hostiles

Cuando el volumen de datos recolectado es bajo, se aplican transformaciones estocásticas en tiempo real sobre el conjunto de entrenamiento para simular las condiciones operativas de una terminal real:

* `RandomRotation` y `RandomResizedCrop`: Simulan billetes insertados de forma torcida o a diferentes distancias del sensor óptico.
* `ColorJitter`: Altera aleatoriamente el brillo, contraste y saturación, emulando la degradación lumínica del entorno (luz artificial vs. luz solar directa).

### 2.5 Calibración de Probabilidades y Umbrales de Seguridad

Una red neuronal con salida `Softmax` entrega una distribución de probabilidad sobre las clases conocidas, pero tiende a ser sobreconfiada, sufriendo de alucinaciones ante datos fuera de distribución (*Out-of-Distribution*).
Para mitigar fallos operativos o intentos de fraude, se introduce un **Umbral de Confianza (Thresholding)**:

$$\text{Decisión} = \begin{cases} \text{Clase}_i & \text{si } P(\text{Clase}_i) \ge \tau \\ \text{Rechazado} & \text{si } P(\text{Clase}_i) < \tau \end{cases}$$

Donde $\tau$ representa el límite mínimo de seguridad técnica (por ejemplo, 85%). Si la máxima probabilidad no supera este umbral, el sistema rechaza la transacción por falta de certeza.

---

## Eje Temático 3: Visión por Computadora en Tiempo Real (Caso: Conteo de Tráfico con YOLO)

El último módulo práctico abordó el procesamiento temporal, transitando de imágenes estáticas individuales a flujos de video continuos donde la localización espacial y la persistencia temporal de los objetos son mandatorias.

### 3.1 Clasificación vs. Detección de Objetos

* **Clasificación:** Asigna una única etiqueta global a toda la imagen. No determina la ubicación espacial de los elementos.
* **Detección de Objetos:** Resuelve simultáneamente dos problemas: **Clasificación** (qué es el objeto) y **Regresión** (cuáles son sus coordenadas espaciales exactas expresadas como una caja delimitadora u *Bounding Box*: $[x_{center}, y_{center}, width, height]$).

### 3.2 La Revolución Arquitectónica de YOLO (You Only Look Once)

Los sistemas de detección antiguos (como R-CNN) utilizaban algoritmos para proponer regiones de interés y luego pasaban cada región por una CNN clasificadora, siendo un proceso lento e inviable para video en tiempo real.

* **Enfoque Single-Shot:** YOLO reformula la detección como un único problema de regresión. Una sola red neuronal convolucional procesa la imagen completa de una sola pasada.
* **Mecánica interna:** Divide la imagen de entrada en una cuadrícula (grid) de celdas. Si el centro de un objeto cae dentro de una celda, esa celda es la encargada de predecir las coordenadas de la caja y las probabilidades de la clase asociada. Esto permite que arquitecturas como YOLOv8 procesen flujos de video a velocidades superiores a los 100 FPS en hardware estándar de GPU.

### 3.3 El Proceso de Rastreo Temporal (Object Tracking)

La detección por sí sola es agnóstica al tiempo; un detector puro ve un auto en el cuadro $t$ y vuelve a ver un auto en el cuadro $t+1$, pero no sabe si es el mismo vehículo o uno nuevo. El *Tracking* soluciona esto acoplando un algoritmo de asociación de datos (como *ByteTRACK*):

* **Centroide:** Se calcula el centro geométrico de la caja delimitadora detectada.
* **Asociación de Identidad:** El algoritmo calcula la distancia geométrica y la métrica de Intersección sobre la Unión (IoU) entre las cajas del cuadro actual y las predicciones basadas en la velocidad y trayectoria del cuadro anterior. Si la correlación supera el umbral matemático, se le asigna un identificador único (ID) que persiste a lo largo del tiempo.

### 3.4 Geometría Espacial del Conteo (Tripwires)

Para implementar un contador automatizado en un entorno vial (como el análisis de flujo de una rotonda urbana), se utiliza la lógica geométrica de líneas de cruce o *Line Zones*:

1. Se define una línea virtual en el espacio de la imagen mediante coordenadas de inicio y fin: $L = \overline{P_1P_2}$.
2. En cada cuadro de video, el sistema evalúa el vector de movimiento generado por el centroide de cada ID único.
3. Utilizando álgebra vectorial, se calcula la intersección entre el segmento de la trayectoria del vehículo y el segmento de la línea virtual.
4. Al comprobarse la intersección, el sistema valida la dirección del movimiento e incrementa de forma irreversible el contador de la clase específica (Auto, Moto, Camión), bloqueando el ID para evitar dobles conteos.

### 3.5 Desafíos de Producción en Entornos Dinámicos

* **Oclusión:** Ocurre cuando un objeto de gran escala (como un colectivo o camión) bloquea visualmente la línea de visión de un objeto más pequeño (como una motocicleta). Al desaparecer el objeto del detector, el rastreador pierde el ID; al reaparecer cuadros después, el sistema suele asignarle un nuevo ID, generando un error de sobreconteo.
* **Perspectiva y Deformación Geométrica:** A medida que los vehículos giran en una rotonda o se alejan de la cámara, su aspecto visual cambia de forma drástica (pasan de una vista frontal a una lateral y luego posterior). El detector debe ser lo suficientemente robusto para mantener una alta confianza de detección a pesar de la rotación en el espacio tridimensional.

---

## Matriz Resumen de Conceptos Transversales

| Concepto Técnico | Función Principal | Contexto de Aplicación Crítica |
| --- | --- | --- |
| **Data Augmentation** | Multiplicar de forma sintética la diversidad del dataset a través de alteraciones estocásticas. | Crucial al trabajar con datasets pequeños recolectados manualmente en campo (ej. Billetes PYG). |
| **Transfer Learning** | Reutilizar el conocimiento de bajo y medio nivel de un modelo entrenado en un dataset masivo. | Necesario para acelerar el tiempo de desarrollo y asegurar convergencia con pocos datos de entrenamiento. |
| **Depthwise Separable Conv.** | Reducir el costo computacional dividiendo la convolución en pasos independientes por canal y por punto. | Mandatorio para el despliegue en sistemas embebidos, terminales de autoservicio y dispositivos Edge. |
| **Single-Shot Detection** | Procesar la localización espacial y la clasificación en una única pasada por la red de extremo a extremo. | Esencial para alcanzar el rendimiento en cuadros por segundo (FPS) exigido por el análisis de video en tiempo real. |
| **Temporal Tracking** | Mantener la persistencia de identidad de un objeto a través del tiempo basándose en distancias y trayectorias. | Requerido para implementar analítica viales, conteo logístico y seguridad automatizada. |
