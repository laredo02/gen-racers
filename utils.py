from time import sleep

import pygame
from PIL import Image
import numpy as np
import math


def on_map(mapa,
           coord):  # matriz de mapa y coordenada (x, y) que se quiere comprobar. devuelve true si la cordenada esta en el track
    return mapa[coord[1]][coord[0]] == 0


def on_checkpoint(checkpoint, coord):  # Comprobar si una coordenada esta dentro de un checkpoint
    for c in checkpoint:
        if c[0] == coord[0] and c[1] == coord[1]:
            return True
    return False


def min_distance_to_checkpoint(coord,
                               checkpoint):  # coordenada (x, y), lista de coordenadas de un checkpoint [(x,y), (x, y), (), ...], devuelve la distancia al pixel mas cercano
    if checkpoint == []:
        return -1
    mindist = 1e10
    for pixel in checkpoint:
        coord_to_checkpoint = (pixel[0] - coord[0], pixel[1] - coord[1])
        dist = math.sqrt(coord_to_checkpoint[0] ** 2 + coord_to_checkpoint[1] ** 2)
        if dist < mindist:
            mindist = dist
    return mindist


def checkpoint_middle_pixel(checkpoint):  # devuelve el pixel central de un checkpoint
    return checkpoint[len(checkpoint) // 2]


def distance_to_center_checkpoint(coord, checkpoint):  # distancia euclidea al centro del checkpoint
    check = checkpoint_middle_pixel(checkpoint)
    coord_to_checkpoint = (check[0] - coord[0], check[1] - coord[1])
    dist = math.sqrt(coord_to_checkpoint[0] ** 2 + coord_to_checkpoint[1] ** 2)
    return dist


def image_to_matrix(image_path, threshold=128):  # Convertir el mapa en una matriz
    image = Image.open(image_path).convert('L')  # Abrir la imagen y convertirla a escala de grises
    image_array = np.array(image)  # Convertir la imagen a un array de numpy
    binary_array = (image_array > threshold).astype(int)  # Umbralizar la imagen para obtener valores binarios
    return binary_array


def extract_blue_checkpoints(image_path, max_checkpoints=256):  # Extraer los checkpoints de la mascara
    image = Image.open(image_path).convert('RGBA')  # Abrir la imagen
    image_array = np.array(image)  # Convertir la imagen a un array de numpy
    checkpoints = [[] for _ in range(max_checkpoints)]  # Crear una lista de listas para almacenar los checkpoints
    for y in range(image_array.shape[0]):  # Iterar sobre los píxeles
        for x in range(image_array.shape[1]):
            pixel = image_array[y, x]
            if len(pixel) == 4:  # RGBA
                r, g, b, a = pixel
            else:  # RGB
                r, g, b = pixel
                a = 255  # Asumir que el alfa es 255 si no está presente
            if r == 0 and b == 255 and a == 255:
                checkpoints[g].append([x, y])
    checkpoints = [chk for chk in checkpoints if chk]
    return checkpoints


def extract_red_checkpoints(image_path, max_checkpoints=256):  # Extraer los checkpoints de la mascara
    image = Image.open(image_path).convert('RGBA')  # Abrir la imagen
    image_array = np.array(image)  # Convertir la imagen a un array de numpy
    checkpoints = [[] for _ in range(max_checkpoints)]  # Crear una lista de listas para almacenar los checkpoints
    for y in range(image_array.shape[0]):  # Iterar sobre los píxeles
        for x in range(image_array.shape[1]):
            pixel = image_array[y, x]
            if len(pixel) == 4:  # RGBA
                r, g, b, a = pixel
            else:  # RGB
                r, g, b = pixel
                a = 255  # Asumir que el alfa es 255 si no está presente
            if r == 255 and b == 0 and a == 255:
                checkpoints[g].append([x, y])
    checkpoints = [chk for chk in checkpoints if chk]
    return checkpoints


def print_map(matrix):  # Imprimir mapa
    for row in matrix:
        for pixel in row:
            print(pixel, end=' ')
        print()  # Nueva línea al final de cada fila


def print_checkpoints(checkpoints):  # Imprimir coordenadas de los pixeles que forman cada checkpoint
    for i, checkpoint in enumerate(checkpoints):
        print(f"Checkpoint {i}: {checkpoint}")


def print_coord_on_track(mapa, blue_checkpoints, red_checkpoints,
                         coord):  # imprime una coordenada (x, y) dentro del mapa y con los checkpoints
    all_blue_coordinates = [coord for checkpoint in blue_checkpoints for coord in checkpoint]
    all_red_coordinates = [coord for checkpoint in red_checkpoints for coord in checkpoint]

    all_blue_coordinates_tuple = []
    for i, j in all_blue_coordinates:
        all_blue_coordinates_tuple.append((i, j))

    all_red_coordinates_tuple = []
    for i, j in all_red_coordinates:
        all_red_coordinates_tuple.append((i, j))

    coord = (coord[0], coord[1])

    i = 0
    for row in mapa:
        j = 0
        for pixel in row:
            if (j, i) == coord:
                print('*', end=' ')
            elif (j, i) in all_blue_coordinates_tuple:
                print('B', end=' ')
            elif (j, i) in all_red_coordinates_tuple:
                print('R', end=' ')
            else:
                print(pixel, end=' ')
            j += 1
        i += 1
        print()


def print_track(mapa, checkpoints):  # Imprimir mapa y checkpionts superpuestos
    all_coordinates = [coord for checkpoint in checkpoints for coord in checkpoint]
    i = 0
    for row in mapa:
        j = 0
        for pixel in row:
            if (j, i) in all_coordinates:
                print(' ', end=' ')
            else:
                print(pixel, end=' ')
            j += 1
        i += 1
        print()  # Nueva línea al final de cada fila


def checkpoint_centre(checkpoint):
    return checkpoint[len(checkpoint) // 2]


# Muestra los caminos de una poblacion
def display_paths(screen, track, genome, colours, posicion_origen):
    screen.blit(track, (0, 0))
    for i in range(len(colours)):
        display_path(screen, genome[i], colours[i])


# Muestra el camino seguido po un individuo en el color provisto
def display_path(screen, genome, track, colour):
    screen.blit(track, (0, 0))
    for gene in genome:
        display_position(screen, gene, colour)


# Muestra el movimiento de un individuo siempre en verde, sin dejar rastro
def display_genome(display, screen, track, genome, ):
    for gene in genome:
        screen.blit(track, (0, 0))
        display_position(screen, gene, (0,255,0))


# Computa la posicion despues de consumir un gen
def move(position, gene):
    movement = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]]
    position[0] = + movement[gene][0]
    position[1] = + movement[gene][1]
    return position


# Cambia el pixel de la pantalla a un color parado por parametro. Esta escalado por 8 para que pueda verse correctamente
def display_position(screen, position, colour):
    for i in range(4):
        for j in range(4):
            screen.set_at((4 * position[0] + i, 4 * position[1] + j), colour)
    pygame.display.flip()


