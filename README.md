# Atividade 05 - Básico em Machine Learning

Este repositório contém um resumo estruturado dos principais conceitos abordados no Módulo 3 do curso Básico em Machine Learning: Pré-processamento, Segmentação e Detecção/Classificação de imagens.

## Índice
1. [Pré-processamento de Imagens](#pré-processamento-de-imagens)
2. [Segmentação de Imagens](#segmentação-de-imagens)
3. [Detecção/Classificação de Imagens](#detecçãoclassificação-de-imagens)

---

## Pré-processamento de Imagens

### Introdução
O pré-processamento de imagens é uma etapa essencial em pipelines de machine learning, garantindo qualidade e uniformidade antes do treinamento. Inclui operações como redimensionamento, normalização, remoção de ruídos e data augmentation.

### Bibliotecas/Frameworks
- **OpenCV**
- **PIL (Python Imaging Library)**
- **TensorFlow e Keras**

### Aplicação Prática
```python
# Redimensionamento e Normalização com OpenCV
import cv2

imagem = cv2.imread('imagem.jpg')
imagem_redimensionada = cv2.resize(imagem, (224, 224))
imagem_normalizada = imagem_redimensionada / 255.0
```
*Este código realiza redimensionamento e normalização padrão para modelos de ML.*

---

## Segmentação de Imagens

### Introdução
A segmentação consiste na divisão da imagem em regiões ou segmentos significativos para facilitar análises, sendo aplicada em imagens médicas, satélites e visão computacional.

### Bibliotecas/Frameworks
- **OpenCV**
- **scikit-image**
- **U-Net (TensorFlow/Keras)**

### Aplicação Prática
```python
# Segmentação com Thresholding usando scikit-image
from skimage import io, filters
import matplotlib.pyplot as plt

imagem = io.imread('imagem.jpg', as_gray=True)
limiar = filters.threshold_otsu(imagem)
segmentada = imagem > limiar

plt.imshow(segmentada, cmap='gray')
plt.show()
```
*Este exemplo segmenta uma imagem usando limiar adaptativo.*

---

## Detecção/Classificação de Imagens

### Introdução
Estas técnicas identificam objetos específicos (detecção) e categorizam imagens (classificação), utilizando principalmente redes neurais profundas.

### Bibliotecas/Frameworks
- **TensorFlow/Keras** (VGG, ResNet, MobileNet)
- **YOLO**
- **PyTorch**

### Aplicação Prática
```python
# Classificação com TensorFlow/Keras (MobileNetV2)
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

modelo = MobileNetV2(weights='imagenet')
img = image.load_img('imagem.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = modelo.predict(x)
print('Previsões:', decode_predictions(preds, top=3)[0])
```
*Este exemplo ilustra o uso do MobileNetV2 para classificação de imagens.*

