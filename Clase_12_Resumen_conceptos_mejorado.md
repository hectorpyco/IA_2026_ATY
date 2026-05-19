# Guía de Consolidación: Visión por Computadora Aplicada
## Del Laboratorio al Sistema en Producción — Material de Repaso para Ingeniería

Este resumen integra los conceptos trabajados en los tres laboratorios del curso: **Inspección Industrial de Metales (NEU)**, **Clasificación de Billetes Guaraníes (PYG)** y **Conteo de Tráfico en Tiempo Real (YOLO)**. El hilo conductor es siempre el mismo: pasar de tablas de números a imágenes, y de imágenes estáticas a video en tiempo real.

> 💡 **Cómo usar esta guía:** Cada sección conecta la teoría con el código que escribieron. Si algo no queda claro, volvé al notebook correspondiente y buscá el bloque mencionado.

---

## Eje 1: Fundamentos del Tensor y Clasificación de Imágenes
### *Caso: Inspección Industrial de Metales — Dataset NEU*

### 1.1 Una imagen no es una foto: es una matriz de números

El primer obstáculo conceptual del curso fue entender que **una red neuronal no "ve" imágenes, procesa números**. Una imagen en escala de grises es una matriz 2D donde cada celda contiene la intensidad de un píxel entre 0 (negro) y 255 (blanco). Una imagen a color es tres matrices apiladas: una por canal (Rojo, Verde, Azul).

```
Imagen 4×4 en escala de grises:
┌─────┬─────┬─────┬─────┐
│  12 │  45 │  89 │ 201 │
├─────┼─────┼─────┼─────┤
│  34 │ 120 │  67 │  90 │
└─────┴─────┴─────┴─────┘
```

### 1.2 El Tensor 4D: cómo PyTorch organiza las imágenes

Para procesar eficientemente en GPU, las imágenes se agrupan en lotes (*batches*) dentro de una estructura de cuatro dimensiones:

$$\text{Shape} = [\text{Batch Size}, \text{Canales}, \text{Alto}, \text{Ancho}]$$

Por ejemplo, un lote de 32 imágenes en escala de grises de 224×224 píxeles produce el tensor `[32, 1, 224, 224]` — es decir, 1.605.632 números procesados simultáneamente. En el caso PYG con imágenes a color: `[16, 3, 224, 224]`.

| Dimensión | En NEU (gris) | En PYG (color) | ¿Para qué sirve? |
|---|---|---|---|
| **Batch** | 32 | 16 | Cuántas imágenes procesamos juntas |
| **Canales** | 1 (gris) | 3 (RGB) | Profundidad del color |
| **Alto × Ancho** | 224 × 224 | 224 × 224 | Resolución espacial |

### 1.3 ¿Por qué una CNN y no una red común (MLP)?

Una red densa trataría cada píxel como un dato independiente, sin considerar que los píxeles **vecinos están relacionados**. Un rasguño en acero es una secuencia de píxeles oscuros *en línea*, no píxeles oscuros aislados. Las Redes Neuronales Convolucionales (CNN) resuelven esto con tres tipos de capas:

**Capa Convolucional (`Conv2d`):** Un filtro (matriz pequeña, típicamente 3×3) se desliza por toda la imagen, calculando el producto punto entre el filtro y cada región local. Cada filtro aprende a detectar un patrón: bordes horizontales, verticales, texturas. Lo importante es que **la red aprende sola los valores de estos filtros durante el entrenamiento**.

**Capa de Pooling (`MaxPool2d`):** Toma regiones de 2×2 píxeles y conserva solo el valor máximo. Resultado: la imagen se reduce a la mitad. Esto hace que la red sea tolerante a pequeñas variaciones de posición — si el rasguño está 2 píxeles más a la izquierda, igual se detecta.

**Capas Densas (`Linear`):** Después de varias rondas de Conv+Pool, el tensor resultante se "aplasta" y se pasa por capas densas que toman la decisión final: ¿a cuál de las 6 clases pertenece esta imagen?

**Jerarquía de lo que aprende cada capa:**
- Capas iniciales → bordes, contrastes, esquinas
- Capas intermedias → texturas, formas simples
- Capas profundas → patrones completos, rasgos semánticos del objeto

### 1.4 Reproducibilidad: por qué siempre fijamos la semilla

Las redes neuronales usan números aleatorios durante la inicialización de pesos y en el Dropout. Sin fijar una semilla, dos ejecuciones del mismo código dan resultados distintos, lo que hace imposible comparar experimentos o depurar errores. Esto es un **estándar profesional** en ciencia de datos:

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# En GPU:
torch.backends.cudnn.deterministic = True
```

### 1.5 El pipeline de transformaciones: de JPEG a tensor

Las redes no trabajan con archivos de imagen: trabajan con tensores normalizados. El pipeline de transformación convierte cada imagen al formato esperado:

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),       # Tamaño fijo para todos los batches
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),   # Aumentación
    transforms.RandomRotation(15),       # Aumentación
    transforms.ToTensor(),               # [0,255] → [0,1]
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # → rango ~[-1,1]
])
```

**Regla importante:** la aumentación va **solo en el set de entrenamiento**. El set de validación siempre recibe la imagen original para medir el rendimiento en condiciones reales.

### 1.6 Estabilización del entrenamiento

Entrenar una red desde cero presenta desafíos de convergencia:

**Batch Normalization:** Normaliza las activaciones de cada capa para cada lote de entrenamiento, mitigando el *Internal Covariate Shift* (la distribución de activaciones cambia constantemente durante el entrenamiento, dificultando la convergencia). Permite usar tasas de aprendizaje más altas.

**Dropout:** "Apaga" aleatoriamente una fracción de neuronas (ej: 30% o 50%) en cada paso de entrenamiento. Fuerza a la red a aprender representaciones redundantes y robustas, combatiendo el sobreajuste directamente.

**Gradient Clipping:** En el Bloque 5 del notebook NEU vieron este patrón:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Impide que los gradientes tomen valores gigantes ("explosión de gradientes"), lo que causaría saltos descontrolados en los pesos.

**Early Stopping:** Monitorea la pérdida de validación. Si no mejora durante N épocas consecutivas, detiene el entrenamiento para prevenir sobreajuste:

```python
if val_loss < mejor_val_loss:
    mejor_val_loss = val_loss
    torch.save(model.state_dict(), 'mejor_modelo.pth')  # Guardar el mejor
    epocas_sin_mejora = 0
else:
    epocas_sin_mejora += 1
    if epocas_sin_mejora >= PATIENCE:
        break  # Detener entrenamiento
```

### 1.7 Diagnóstico con curvas de aprendizaje

Las curvas de pérdida y accuracy por época son el principal instrumento de diagnóstico de un modelo:

| Lo que ves en las curvas | Diagnóstico | Qué hacer |
|---|---|---|
| Pérdida alta en train y val | **Underfitting** (modelo demasiado simple) | Más épocas, arquitectura más profunda |
| Train baja, val estancada o sube | **Overfitting** (memorización) | Más Dropout, más Data Augmentation, Early Stopping |
| Ambas bajan juntas y convergen | **Entrenamiento correcto** | Continuar y evaluar |

### 1.8 Métricas correctas para inspección industrial

En control de calidad, la métrica de *Accuracy* (precisión global) es engañosa si hay desbalance de clases. Las métricas basadas en la Matriz de Confusión son más informativas:

- **Precision:** De todo lo que el modelo dijo "es defecto X", ¿cuánto era realmente defecto X? → Penaliza falsos positivos.
- **Recall:** De todos los defectos X reales, ¿cuántos detectó el modelo? → Penaliza falsos negativos.
- **F1-Score:** Media armónica entre Precision y Recall.

**En inspección industrial, el Recall es la métrica crítica.** Es más grave dejar pasar un defecto (falso negativo) que rechazar una pieza buena (falso positivo). Un defecto que llega al cliente puede causar fallas estructurales; una pieza buena descartada solo es un costo extra de producción.

### 1.9 El Fenómeno del Domain Shift

*Domain Shift* ocurre cuando la distribución estadística de los datos de entrenamiento difiere de los datos en producción real. En el notebook NEU lo experimentaron en la tarea: un modelo con 95%+ de accuracy en el dataset de laboratorio puede fallar con fotos de internet del mismo tipo de defecto.

¿Por qué? El dataset NEU fue capturado con un equipo específico de microscopía bajo condiciones controladas. Un cambio en la iluminación, la resolución, el ángulo de captura o los artefactos de compresión JPEG puede confundir al modelo. Este es **el mayor problema del ML en producción**.

---

### Tabla de conceptos clave — Eje 1

| Concepto | ¿Qué es? | ¿Por qué importa? |
|---|---|---|
| **Tensor 4D** | `[Batch, Canal, H, W]` | Formato estándar de imágenes en PyTorch |
| **Convolución** | Filtro que detecta patrones locales | El núcleo de toda CNN |
| **BatchNorm** | Normaliza activaciones por batch | Estabiliza y acelera el entrenamiento |
| **Dropout** | Apaga neuronas aleatoriamente | Combate el overfitting |
| **Data Augmentation** | Transforma imágenes al azar en cada época | Genera variantes sintéticas, mejora generalización |
| **Early Stopping** | Detiene si val_loss no mejora N épocas | Guarda el mejor modelo, previene overfitting |
| **F1-Score** | Media armónica de Precision y Recall | Métrica balanceada para datasets industriales |
| **Domain Shift** | Diferencia entre datos de entrenamiento y producción | Causa principal de fallos en implementaciones reales |

---

## Eje 2: El Paradigma Data-Centric y Eficiencia en el Borde
### *Caso: Clasificador de Billetes Guaraníes — PYG*

### 2.1 El debate Model-Centric vs. Data-Centric

El notebook de billetes introdujo un cambio de filosofía fundamental respecto al caso NEU:

**Model-Centric:** ¿El modelo comete errores? Hacelo más grande o cambiá la arquitectura.

**Data-Centric (Andrew Ng):** ¿El modelo comete errores? Mejorá los datos primero.

En la mayoría de los proyectos industriales reales, mejorar los datos supera en impacto a mejorar el modelo. En el clasificador de billetes, la robustez no proviene de usar una red más compleja, sino de recolectar datos con **toda la variabilidad real** que enfrentará la terminal:

| Desafío real de la terminal | ¿Cómo lo capturaron en el dataset? |
|---|---|
| Billete desgastado vs. nuevo | Fotografiar billetes viejos y arrugados |
| Iluminación variable | Fotos con luz fluorescente, natural, en sombra |
| Billete insertado torcido | Rotaciones de 15°, 45°, 90° |
| Dedos del usuario visibles | Sostener el billete con dedos en cuadro |
| Distintas cámaras del grupo | Cada integrante fotografía con su celular |

> **"Garbage In, Garbage Out":** Si todas las fotos de entrenamiento muestran el billete perfectamente centrado sobre fondo blanco bajo luz de ventana, la terminal fallará en cualquier otra condición. La diversidad del dataset determina la robustez del sistema.

### 2.2 Transfer Learning: reutilizar conocimiento previo

Entrenar una red profunda desde cero requiere millones de imágenes y días de cómputo. El *Transfer Learning* resuelve esto reutilizando un modelo ya entrenado.

**Analogía:** Es como contratar a un médico especialista en radiología para que aprenda a leer tomografías de tórax. No lo entrenás desde cero en anatomía básica: ya sabe interpretar imágenes médicas, ya reconoce anomalías. Solo necesita aprender las especificidades del nuevo tipo de imagen.

**MobileNetV2** fue entrenado por Google durante semanas en 1.4 millones de imágenes (ImageNet). Ya aprendió bordes, texturas, formas y colores. Solo le enseñamos qué combinación de esas características corresponde a cada denominación del Guaraní.

**Estrategia: Feature Extraction**

```python
modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# 1. Congelar todas las capas convolucionales
for param in modelo.parameters():
    param.requires_grad = False  # PyTorch: "no calculés gradientes aquí"

# 2. Reemplazar solo la capa de clasificación final
modelo.classifier[1] = nn.Sequential(
    nn.Linear(1280, 512),   # 1280 = características que entrega MobileNet
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(512, NUM_CLASSES)  # Solo estos pesos se entrenan
)
```

El esquema es:
```
MobileNetV2 (congelada)
├── Capas convolucionales  ← "Los ojos" — no modificar
│   Ya detectan: bordes, texturas, colores, formas
└── Clasificador original (1000 clases ImageNet)  ← Reemplazar
         ↓
    Nuevo clasificador para billetes PYG
    ├── Linear(1280 → 512)
    ├── ReLU + Dropout(0.4)
    └── Linear(512 → 6 denominaciones)
```

### 2.3 ¿Por qué los valores de normalización son esos?

En el notebook PYG vieron estos números específicos:

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

No son arbitrarios: son las estadísticas calculadas sobre todo el dataset ImageNet. Como MobileNetV2 fue entrenada con esas estadísticas, nuestras imágenes deben estar en el mismo rango de valores. Es como hablar el mismo "idioma numérico" que ya conoce la red.

### 2.4 Arquitectura eficiente: MobileNetV2 y las Convoluciones Separables en Profundidad

Para una terminal de autoservicio, la velocidad de inferencia importa tanto como la precisión. MobileNetV2 es entre 5x y 10x más rápida que ResNet-50 con pérdida mínima de precisión, gracias a las **Convoluciones Separables en Profundidad (Depthwise Separable Convolutions)**:

Una convolución estándar de 3×3 sobre K canales realiza todas las operaciones a la vez. La versión separable la divide en dos pasos independientes: una convolución por canal (*depthwise*) seguida de una convolución puntual 1×1 (*pointwise*). Esto reduce drásticamente el número de parámetros y operaciones de punto flotante (FLOPs), permitiendo inferencias en milisegundos en hardware con recursos limitados.

### 2.5 Data Augmentation avanzada para entornos hostiles

Con 15 fotos por clase, el riesgo de que la red memorice los datos exactos es alto. La aumentación de datos aplica transformaciones aleatorias en cada época para simular las condiciones reales de la terminal:

```python
transform_train = transforms.Compose([
    transforms.RandomRotation(degrees=30),         # Billete insertado torcido
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)), # Zoom variable
    transforms.RandomHorizontalFlip(p=0.5),        # Cara del billete invertida
    transforms.ColorJitter(brightness=0.4, contrast=0.4), # Iluminación variable
    transforms.RandomGrayscale(p=0.05),            # Sombras o manchas
])
```

Cada parámetro responde a un escenario físico real. El objetivo es que el modelo aprenda el billete en sí, no las condiciones de la foto.

### 2.6 Umbral de confianza: evitar el billete equivocado

Una red con salida `Softmax` siempre entrega una distribución de probabilidades sobre las clases conocidas — incluso si le mostramos un papel arrugado, el modelo asignará probabilidades a cada denominación. Para evitar aceptar una transacción con baja certeza, se introduce un umbral de confianza:

$$\text{Decisión} = \begin{cases} \text{Clase}_i & \text{si } P(\text{Clase}_i) \ge \tau \\ \text{Rechazado} & \text{si } P(\text{Clase}_i) < \tau \end{cases}$$

Donde $\tau$ es el umbral mínimo de seguridad (por ejemplo, 85%). En el Bloque 9 del notebook, el "Simulador de terminal de depósito" implementa exactamente esta lógica: si la confianza no supera el umbral, la terminal rechaza la transacción en lugar de arriesgarse.

---

### Tabla de conceptos clave — Eje 2

| Concepto | Función Principal | Contexto práctico |
|---|---|---|
| **Data-Centric AI** | Enfocar los esfuerzos en los datos, no en la arquitectura | La robustez del clasificador de billetes vino de los datos, no del modelo |
| **Transfer Learning** | Reutilizar conocimiento de un modelo pre-entrenado en dataset masivo | Nos permite entrenar con ~90 fotos en vez de millones |
| **Feature Extraction** | Congelar el backbone y solo entrenar la cabeza de clasificación | `requires_grad = False` en todas las capas convolucionales |
| **Depthwise Separable Conv.** | Reducir FLOPs dividiendo la convolución en dos pasos | Hace posible inferencia en tiempo real en hardware de bajo costo |
| **ImageNet Stats** | Mean y std de 1.4M imágenes, necesarios para TL | `[0.485, 0.456, 0.406]` para mean; `[0.229, 0.224, 0.225]` para std |
| **Umbral de confianza** | Rechazar predicciones con baja certeza | Seguridad operativa de la terminal |

---

## Eje 3: Visión por Computadora en Tiempo Real
### *Caso: Conteo de Tráfico con YOLOv8*

### 3.1 El salto: de imágenes estáticas a video

Los dos ejes anteriores trabajaban con imágenes individuales. El video agrega una dimensión nueva: **el tiempo**. Una moto a 60 km/h recorre 16.7 metros por segundo. Si la zona visible de la cámara abarca 10 metros, la moto estará en cuadro **0.6 segundos**. A 10 FPS, eso son solo 6 cuadros para detectar, rastrear y contar.

**Clasificación vs. Detección de Objetos:**

| Clasificación (NEU, PYG) | Detección (YOLO) |
|---|---|
| Asigna una etiqueta a toda la imagen | Resuelve simultáneamente: ¿qué hay? y ¿dónde está? |
| Salida: una clase | Salida: clase + coordenadas de bounding box `[x, y, w, h]` |
| Una imagen = una decisión | Una imagen = múltiples detecciones posibles |

### 3.2 La revolución de YOLO: Single-Shot Detection

Los sistemas anteriores (R-CNN) proponían regiones candidatas y luego clasificaban cada una por separado — un proceso de dos etapas, lento e inviable para video.

**YOLO (You Only Look Once)** reformula la detección como un único problema de regresión. Una sola red neuronal convolucional procesa la imagen completa en una sola pasada. Internamente, divide la imagen en una cuadrícula de celdas. Si el centro de un objeto cae dentro de una celda, esa celda predice las coordenadas de la caja y la probabilidad de la clase.

YOLOv8 ofrece cinco variantes según el hardware disponible:

| Modelo | Archivo | mAP | FPS (GPU) | Uso recomendado |
|---|---|---|---|---|
| Nano | `yolov8n.pt` | 37.3 | ~160 | Raspberry Pi, CPU sin GPU |
| Small | `yolov8s.pt` | 44.9 | ~120 | Laptops con GPU básica |
| Medium | `yolov8m.pt` | 50.2 | ~80 | Workstations |
| Large | `yolov8l.pt` | 52.9 | ~50 | Servidores |
| XLarge | `yolov8x.pt` | 53.9 | ~30 | Máxima precisión |

> **En clase usamos `yolov8n.pt`** para que funcione en CPU. Con GPU disponible, `yolov8m.pt` mejora la detección de motos en ángulos complicados.

### 3.3 El problema de identidad: por qué necesitamos Tracking

La detección por sí sola es agnóstica al tiempo. Un detector puro ve un auto en el cuadro $t$ y vuelve a ver un auto en el cuadro $t+1$, pero no sabe si es el mismo o uno nuevo. Sin tracking, el mismo auto podría ser contado decenas de veces.

**ByteTrack** resuelve esto asignando un ID único a cada objeto y manteniéndolo entre cuadros. El flujo completo por cada cuadro de video es:

```
Cuadro de video
      ↓
  YOLOv8 → Lista de detecciones: [caja, clase, confianza]
      ↓
  Filtro de clases (car=2, moto=3, bus=5, truck=7)
      ↓
  ByteTrack → Asigna tracker_id único y persistente
      ↓
  LineZone → Si el centroide cruzó la línea → suma al contador
      ↓
  Anotadores → Dibuja cajas, etiquetas y línea sobre el cuadro
      ↓
  Salida (ventana / archivo de video)
```

### 3.4 Cómo funciona la asociación de identidades

Para decidir si el auto detectado en el cuadro actual es el mismo que el del cuadro anterior, ByteTrack utiliza dos métricas:

**Distancia de centroide:** se calcula el centro geométrico de cada bounding box. Si el centroide del cuadro actual está cerca del centroide predicho del cuadro anterior (considerando la velocidad estimada del objeto), probablemente es el mismo vehículo.

**IoU (Intersección sobre la Unión):** mide cuánto se superponen dos cajas delimitadoras. Si dos cajas de cuadros consecutivos se superponen significativamente (IoU alto), se asocian al mismo ID.

```python
tracker = sv.ByteTrack(
    track_thresh=0.35,  # Confianza mínima para iniciar un nuevo track
    track_buffer=30     # Cuadros de espera antes de eliminar un track perdido
)
```

`track_buffer=30` significa que si un vehículo desaparece del detector (por oclusión, por ejemplo) durante hasta 30 cuadros (~1 segundo a 30 FPS), el sistema lo "recuerda" y le reasigna el mismo ID cuando reaparece.

### 3.5 Líneas de cruce (Tripwires): el mecanismo de conteo

Para no contar un vehículo que está parado en el campo de visión, el conteo solo ocurre cuando el vehículo **cruza una línea virtual**:

```python
# Definir la línea en coordenadas de imagen
linea_inicio = sv.Point(0, 400)       # Extremo izquierdo en Y=400
linea_fin    = sv.Point(1280, 400)    # Extremo derecho en Y=400
line_zone    = sv.LineZone(start=linea_inicio, end=linea_fin)

# En cada cuadro:
cruzo_entrando, cruzo_saliendo = line_zone.trigger(detections=detecciones)
# cruzo_entrando[i] == True → el vehículo i cruzó en dirección de entrada
```

El algoritmo verifica la intersección entre la trayectoria del centroide del vehículo (el segmento desde su posición anterior hasta la actual) y la línea virtual. Cuando detecta intersección, valida la dirección e incrementa el contador de forma irreversible para ese ID.

### 3.6 Benchmark: ¿el sistema es suficientemente rápido?

La función `medir_fps()` del Bloque 7 del notebook mide el rendimiento real del sistema en el hardware disponible:

| FPS medidos | Evaluación |
|---|---|
| ≥ 25 FPS | Suficiente para análisis de tráfico en tiempo real |
| 10–24 FPS | Marginal. Puede perder motos a alta velocidad |
| < 10 FPS | Insuficiente. Cambiar a modelo Nano o agregar GPU |

### 3.7 Limitaciones reales del sistema (importante para proyectos de ingeniería)

**Oclusión:** Si un camión tapa una moto, el detector pierde la moto. Al reaparecer, ByteTrack puede reasignar un ID diferente, generando un error de sobreconteo. El parámetro `track_buffer` mitiga esto pero no lo elimina.

**Objetos pequeños o lejanos:** El modelo Nano tiene dificultades con objetos que ocupan menos de ~20×20 píxeles en el cuadro. Una moto lejana puede no ser detectada.

**Clases fijas:** YOLOv8 pre-entrenado solo detecta las 80 clases del dataset COCO. Una mototaxi, una carreta o un tractor agrícola no están incluidos. Para detectarlos, se necesitaría *fine-tuning* con un dataset propio.

**Perspectiva y ángulo:** El modelo fue entrenado mayormente con vistas frontales y laterales. Vistas cenitales (desde un dron o cámara cenital en rotonda) reducen la precisión.

> **Regla de oro:** Para un proyecto real de ingeniería de tráfico, YOLOv8 es un excelente punto de partida, pero siempre debe ser validado con datos reales del sitio y complementado con lógica de negocio específica (horarios, tipos de vehículos locales, condiciones climáticas).

---

### Tabla de conceptos clave — Eje 3

| Concepto | Función | Contexto práctico |
|---|---|---|
| **Single-Shot Detection** | Localización + clasificación en una sola pasada por la red | YOLO puede superar 100 FPS en GPU estándar |
| **Bounding Box** | `[x_centro, y_centro, ancho, alto]` en coordenadas de imagen | Salida de cada detección de YOLO |
| **ByteTrack** | Asigna IDs únicos y persistentes entre cuadros | Sin tracking, el mismo auto se cuenta múltiples veces |
| **IoU** | Intersección sobre la Unión entre dos cajas | Métrica de solapamiento para asociar detecciones entre cuadros |
| **track_buffer** | Cuadros de "memoria" del tracker ante oclusión | `track_buffer=30` ≈ 1 segundo de tolerancia a 30 FPS |
| **LineZone (Tripwire)** | Línea virtual que activa el contador al ser cruzada | Evita contar vehículos detenidos en el campo de visión |
| **Oclusión** | Un objeto tapa a otro en la imagen | Principal causa de errores de sobreconteo en tráfico real |

---

## Matriz Resumen de Conceptos Transversales

Esta tabla recorre los conceptos que aparecen en los tres laboratorios y conecta la teoría con la práctica:

| Concepto | Función Principal | Dónde lo aplicaron | Por qué importa en producción |
|---|---|---|---|
| **Tensor 4D** | Estructura de datos estándar para imágenes en PyTorch | NEU (gris), PYG (color), YOLO | Toda la cadena de procesamiento asume este formato |
| **Data Augmentation** | Multiplicar artificialmente la diversidad del dataset | NEU (aumentación moderada), PYG (aumentación agresiva) | Crítico con datasets pequeños recolectados manualmente |
| **Transfer Learning** | Reutilizar conocimiento de un modelo masivo | PYG con MobileNetV2 pre-entrenado en ImageNet | Permite entrenar con ~90 fotos en lugar de millones |
| **Depthwise Separable Conv.** | Reducir parámetros y FLOPs para hardware limitado | MobileNetV2 en PYG | Mandatorio para terminales embebidas y dispositivos Edge |
| **Early Stopping** | Detener cuando val_loss no mejora | NEU y PYG (Bloque de entrenamiento) | Guarda el mejor modelo, evita sobreajuste automáticamente |
| **Umbral de confianza** | Rechazar predicciones con baja certeza | PYG (Simulador de terminal) | Seguridad operativa ante inputs fuera de distribución |
| **Single-Shot Detection** | Localización y clasificación en una sola pasada | YOLO en el contador de tráfico | Única forma de alcanzar >25 FPS en video en tiempo real |
| **Object Tracking** | Persistencia de identidad entre cuadros de video | ByteTrack en YOLO | Sin esto, el mismo objeto se cuenta múltiples veces |
| **Domain Shift** | Diferencia entre datos de entrenamiento y producción | Tarea del NEU (fotos de internet) | La causa más frecuente de fallos en sistemas reales |
| **F1-Score / Recall** | Métricas correctas en presencia de desbalance de clases | Evaluación en NEU y PYG | Accuracy sola puede ser engañosa en control de calidad |
| **Reproducibilidad (seed)** | Resultados consistentes entre ejecuciones | Bloque 0 de los tres notebooks | Estándar profesional para comparar experimentos |

---

## Guía rápida de diagnóstico

Cuando algo no funciona, seguí este árbol antes de cambiar la arquitectura:

**El modelo no aprende (pérdida no baja):**
→ ¿Fijaste la semilla? ¿El device es correcto (todos los tensores en el mismo lugar)? ¿El learning rate es demasiado alto o bajo?

**Overfitting (train baja, val se estanca):**
→ Más Dropout, más Data Augmentation, Early Stopping, reducir complejidad del modelo.

**Underfitting (ambas curvas con pérdida alta):**
→ Más épocas, arquitectura más profunda, verificar que los datos estén normalizados correctamente.

**Buena accuracy en validación, mala en producción:**
→ Domain Shift. El dataset de entrenamiento no captura la variabilidad real. Recolectar más datos en condiciones de producción.

**El sistema YOLO cuenta mal en video:**
→ Revisar `track_buffer` (¿muy bajo ante oclusiones?), umbral de confianza `CONFIANZA_MINIMA` (¿demasiado bajo genera falsos positivos?), posición de la `LineZone` (¿colocada donde los autos se detienen?).

---

*Curso de Programación Avanzada — FCyT UNCA | Material de preparación para el examen*
