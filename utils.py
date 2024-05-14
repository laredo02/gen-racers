
from PIL import Image
import numpy as np
import math

def on_map(mapa, coord): # matriz de mapa y coordenada (x, y) que se quiere comprobar
    return mapa[coord[1]][coord[0]] == 0

def on_checkpoint(checkpoint, coord):
    return coord in checkpoint

def min_distance_to_checkpoint(coord, checkpoint): # coordenada (x, y), lista de coordenadas de un checkpoint [(x,y), (x, y), (), ...]
    if checkpoint == []:
        return -1
    mindist = 1e10
    for pixel in checkpoint:
        coord_to_checkpoint = (pixel[0] - coord[0], pixel[1] - coord[1])
        dist = math.sqrt(coord_to_checkpoint[0]**2 + coord_to_checkpoint[1]**2)
        if dist < mindist:
            mindist = dist
    return mindist


def image_to_matrix(image_path, threshold=128): # Convertir el mapa en una matriz
    image = Image.open(image_path).convert('L') # Abrir la imagen y convertirla a escala de grises
    image_array = np.array(image) # Convertir la imagen a un array de numpy
    binary_array = (image_array > threshold).astype(int) # Umbralizar la imagen para obtener valores binarios
    return binary_array

def extract_checkpoints(image_path, max_checkpoints=256): # Extraer los checkpoints de la mascara
    image = Image.open(image_path).convert('RGBA')      # Abrir la imagen
    image_array = np.array(image)                       # Convertir la imagen a un array de numpy
    checkpoints = [[] for _ in range(max_checkpoints)]  # Crear una lista de listas para almacenar los checkpoints
    for y in range(image_array.shape[0]): # Iterar sobre los píxeles
        for x in range(image_array.shape[1]):
            pixel = image_array[y, x]
            if len(pixel) == 4:  # RGBA
                r, g, b, a = pixel
            else:  # RGB
                r, g, b = pixel
                a = 255  # Asumir que el alfa es 255 si no está presente
            if r == 255 and b == 255 and a == 255:
                checkpoints[g].append((x, y))
    # Filtrar las listas vacías
    checkpoints = [chk for chk in checkpoints if chk]
    return checkpoints

def print_map(matrix): # Imprimir mapa
    for row in matrix:
        for pixel in row:
            print(pixel, end=' ')
        print()  # Nueva línea al final de cada fila

def print_checkpoints(checkpoints): # Imprimir coordenadas de los pixeles que forman cada checkpoint
    for i, checkpoint in enumerate(checkpoints):
        print(f"Checkpoint {i}: {checkpoint}")

def print_coord_on_track(mapa, checkpoints, coord): # imprime una coordenada (x, y) dentro del mapa
    all_coordinates = [coord for checkpoint in checkpoints for coord in checkpoint]
    i = 0
    for row in mapa:
        j = 0
        for pixel in row:
            if (j, i) == coord:
                print('*', end=' ')
            elif (j, i) in all_coordinates:
                print(' ', end=' ')
            else:
                print(pixel, end=' ')
            j+=1
        i+=1
        print()  # Nueva línea al final de cada fila


def print_track(mapa, checkpoints): # Imprimir mapa y checkpionts superpuestos
    all_coordinates = [coord for checkpoint in checkpoints for coord in checkpoint]
    i = 0
    for row in mapa:
        j = 0
        for pixel in row:
            if (j, i) in all_coordinates:
                print(' ', end=' ')
            else:
                print(pixel, end=' ')
            j+=1
        i+=1
        print()  # Nueva línea al final de cada fila


def checkpoint_centre(checkpoint):
    return checkpoint[len(checkpoint)//2]


# def display_cars(): # interfaz grafica con los caminos de los coches recibe la imagen del mapa, de los checkpoints y una lista de caminos de coches
