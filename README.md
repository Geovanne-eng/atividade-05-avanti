# 📚 Atividade 05 - Básico em Machine Learning

Este repositório contém um resumo detalhado dos principais conceitos abordados no **Módulo 3** do curso **Básico em Machine Learning**. O conteúdo é dividido em três áreas principais: Pré-processamento, Segmentação e Detecção/Classificação de imagens.

---

## 🔖 Índice
- [Pré-processamento de Imagens](#pré-processamento-de-imagens)
- [Segmentação de Imagens](#segmentação-de-imagens)
- [Detecção/Classificação de Imagens](#detecçãoclassificação-de-imagens)

---

## 🖼️ Pré-processamento de Imagens

### Introdução
O pré-processamento de imagens é uma etapa essencial no pipeline de machine learning. Seu principal objetivo é preparar as imagens para treinamento, garantindo que estejam livres de ruídos e padronizadas, aumentando a eficiência e precisão dos modelos.

### Técnicas Detalhadas

- **Redimensionamento**: Ajusta as imagens para dimensões fixas, garantindo que todas as entradas tenham o mesmo tamanho, o que é essencial para redes neurais convolucionais.
- **Normalização**: Transforma os valores dos pixels em uma escala uniforme (geralmente entre 0 e 1), facilitando o aprendizado das redes neurais.
- **Remoção de Ruídos**: Aplicação de filtros como Gaussianos ou medianos para reduzir interferências e melhorar a clareza das imagens.
- **Data Augmentation**: Técnica que expande o conjunto de dados através de modificações como rotações, cortes, espelhamentos e ajustes de brilho, ajudando a evitar o overfitting.

### Bibliotecas Utilizadas
- **OpenCV**
- **PIL (Python Imaging Library)**
- **TensorFlow/Keras**

### Exemplo Prático
```python
import cv2

imagem = cv2.imread('imagem.jpg')
imagem_redimensionada = cv2.resize(imagem, (224, 224))
imagem_normalizada = imagem_redimensionada / 255.0
```

---

## 🎨 Segmentação de Imagens

### Introdução
Segmentação é o processo de dividir uma imagem em regiões distintas, identificando áreas específicas para análise detalhada. É crucial em aplicações como imagens médicas, reconhecimento facial e análise ambiental.

### Técnicas Detalhadas

- **Thresholding**: Divide a imagem com base em valores limiares específicos ou adaptativos.
- **Watershed**: Considera a imagem como um relevo topográfico para segmentar regiões por linhas de divisão naturais.
- **Segmentação baseada em bordas**: Utiliza mudanças de intensidade para identificar e delimitar contornos.
- **Segmentação baseada em regiões**: Agrupa pixels semelhantes para formar regiões homogêneas.

### Bibliotecas Utilizadas
- **OpenCV**
- **scikit-image**
- **U-Net (TensorFlow/Keras)**

### Exemplo Prático
```python
from skimage import io, filters
import matplotlib.pyplot as plt

imagem = io.imread('imagem.jpg', as_gray=True)
limiar = filters.threshold_otsu(imagem)
segmentada = imagem > limiar

plt.imshow(segmentada, cmap='gray')
plt.show()
```

---

## 🎯 Detecção/Classificação de Imagens

### Introdução
Classificação e detecção são técnicas utilizadas para identificar o conteúdo de uma imagem. A classificação atribui categorias gerais à imagem inteira, enquanto a detecção localiza e classifica objetos específicos dentro dela.

### Técnicas Detalhadas

- **Classificação de Imagens**: Categoriza a imagem como um todo com base em características aprendidas.
- **Detecção de Objetos**: Localiza objetos específicos usando técnicas como caixas delimitadoras (bounding boxes).
- **Redes Convolucionais (CNN)**: Redes especializadas para extração e aprendizado de características visuais.
- **YOLO (You Only Look Once)**: Técnica que realiza detecção e classificação simultaneamente, eficiente para aplicações em tempo real.

### Bibliotecas Utilizadas
- **TensorFlow/Keras** (VGG, ResNet, MobileNet)
- **YOLO**
- **PyTorch**

### Exemplo Prático
```python
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

---

📌 Este material visa fornecer um entendimento completo e claro sobre técnicas essenciais em Machine Learning relacionadas a imagens, facilitando o aprendizado e aplicação prática dos conceitos.

