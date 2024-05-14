from PIL import Image
import numpy as np

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


# def display_cars(): # interfaz grafica con los caminos de los coches recibe la imagen del mapa, de los checkpoints y una lista de caminos de coches



