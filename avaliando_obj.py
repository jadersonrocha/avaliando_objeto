import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

# Carregar o modelo ResNet50 pré-treinado (sem a camada fully connected)
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Função para carregar e pré-processar uma imagem
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Redimensiona para o tamanho esperado pela ResNet50
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Função para extrair características de uma imagem
def extract_features(img_array):
    features = model.predict(img_array)
    return features.flatten()

# Função para calcular a similaridade entre duas imagens
def calculate_similarity(features1, features2):
    return cosine_similarity([features1], [features2])[0][0]

# Função para carregar todas as imagens de um diretório e extrair características
def load_images_from_folder(folder):
    features_dict = {}
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img_array = load_and_preprocess_image(img_path)
        features = extract_features(img_array)
        features_dict[filename] = features
    return features_dict

# Função para recomendar imagens similares
def recommend_similar_images(query_img_path, features_dict, top_n=5):
    # Extrair características da imagem de consulta
    query_img_array = load_and_preprocess_image(query_img_path)
    query_features = extract_features(query_img_array)

    # Calcular similaridade com todas as imagens no diretório
    similarities = {}
    for filename, features in features_dict.items():
        similarity = calculate_similarity(query_features, features)
        similarities[filename] = similarity

    # Ordenar por similaridade e retornar as top_n imagens mais similares
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:top_n]

# Função para exibir as imagens recomendadas
def display_recommendations(query_img_path, recommendations, folder):
    query_img = image.load_img(query_img_path)
    plt.figure(figsize=(10, 5))

    # Exibir a imagem de consulta
    plt.subplot(1, len(recommendations) + 1, 1)
    plt.imshow(query_img)
    plt.title("Consulta")
    plt.axis('off')

    # Exibir as imagens recomendadas
    for i, (filename, similarity) in enumerate(recommendations):
        img_path = os.path.join(folder, filename)
        img = image.load_img(img_path)
        plt.subplot(1, len(recommendations) + 1, i + 2)
        plt.imshow(img)
        plt.title(f"Similaridade: {similarity:.2f}")
        plt.axis('off')

    plt.show()

# Diretório contendo as imagens do dataset
dataset_folder = "/img/relogio_1.jpg"

# Carregar todas as imagens e extrair características
features_dict = load_images_from_folder(dataset_folder)

# Caminho para a imagem de consulta
query_img_path = "/img/relogio_1.jpg"

# Recomendar imagens similares
recommendations = recommend_similar_images(query_img_path, features_dict, top_n=5)

# Exibir as recomendações
display_recommendations(query_img_path, recommendations, dataset_folder)
