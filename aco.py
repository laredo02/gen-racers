from copy import deepcopy

import pygame
from utils import *
import numpy as np
import matplotlib.pyplot as plt


def plot_fitness(fitness_values):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot fitness values
    ax.plot(fitness_values, marker='o', linestyle='-')

    # Set labels and title
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Over Generations')

    # Display the plot
    plt.show()

class ACO:

    def __init__(self, map_path: str, blue_checkpoints_path: str, red_checkpoints_path: str, sim_steps: int, n_ants: int, alpha: float = 1, beta: float = 50, rho: float = 0.01):
        self.mapa = image_to_matrix(map_path)                                   # Mapa del track
        self.blue_checkpoints = extract_blue_checkpoints(blue_checkpoints_path) # Checkpoints azules
        self.red_checkpoints = extract_red_checkpoints(red_checkpoints_path)    # Checkpoints rojos
        self.last_checkpoint = len(self.red_checkpoints)-1
        self.coord_inicial = checkpoint_middle_pixel(self.red_checkpoints[-1])  # Coordenada en la que empiezan todos los coches
        self.sim_steps = sim_steps          # Pasos de la simulacion, es decir, numero de coordenadas de cada genoma
        self.n_ants = n_ants                # N de hormigas por iteracion
        self.alpha = alpha                  # Influencia de la feromona
        self.beta = beta                    # Influencia de la heuristica
        self.rho = rho                      # Evaporacion
        self.pheromone = np.ones((len(self.mapa), len(self.mapa[0])))
        self.pheromone_history = []         # Matriz bidimensional de feromonas (mapa de feromonas)
        self.trails_history = []            # Lista de Listas de coordenadas (lista de caminos recorridos por hormigas)
        self.best_fitness_history = []      # Historial de bejor fitnes de cada iteracion
        self.best_solution = None           # Mejor solucion hasta la fecha
        self.best_fitness = float('-inf')   # Fitness de la mejor solucion hasta la fecha

        print(self.red_checkpoints)


        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("racing track")
        self.track_image = pygame.transform.scale(pygame.image.load(map_path), (800, 800))


    def optimize(self, max_evaluations: int = 1000):
        n_evaluations = 0
        while n_evaluations < max_evaluations:
            trails = []
            print("epoch", n_evaluations)
            fitness_list = []
            for _ in range(self.n_ants):
                solution, checkpoints_pasados = self._construct_solution()
                display_path(self.screen, solution, self.track_image, (255, 0, 255))
                fitness = self._evaluate(solution, checkpoints_pasados)
                fitness_list.append(fitness)
                trails.append((solution, fitness))
                if self.best_fitness < fitness:
                    self.best_solution = solution
                    self.best_fitness = fitness

            self.pheromone_history.append(self.pheromone.copy())
            self._update_pheromone(trails, self.best_fitness)
            self.trails_history.append(deepcopy(trails))
            self.best_fitness_history.append(self.best_fitness)
            n_evaluations += 1
            print(fitness)
            print(self.best_fitness)

        plot_fitness(self.best_fitness_history)
        return self.best_solution, self.best_fitness


    def _construct_solution(self):
        solution = [self.coord_inicial]
        next_checkpoint = 0
        step = 0
        prev_blue = False
        prev_red = True
        checkpoints_pasados = 0
        while step < self.sim_steps:
            coord_actual = solution[-1]
            candidates = self._get_candidates(coord_actual) # Candidatos (coordenadas accesibles y que estan dentro del track)
            pheromones = []
            heuristic = []
            for candidate in candidates:
                pheromones.append(self.pheromone[candidate[1]][candidate[0]]**self.alpha)
                heuristic.append(self._heuristic(candidate, next_checkpoint)**self.beta)
            pheromones = np.array(pheromones)
            heuristic = np.array(heuristic)
            candidates = np.array(candidates)
            total = np.sum(pheromones * heuristic)
            probabilities = (pheromones * heuristic) / total
            choices = np.array(range(len(candidates)))
            selected_coord = np.random.choice(choices, p=probabilities.flatten())
            selected_coord = candidates[selected_coord]
            on_red = on_checkpoint(self.red_checkpoints[next_checkpoint], selected_coord)
            on_blue = on_checkpoint(self.blue_checkpoints[next_checkpoint], selected_coord)
            if prev_blue:
                if on_red:
                    if next_checkpoint != self.last_checkpoint:
                        next_checkpoint += 1
                    else:
                        next_checkpoint = 0
                    checkpoints_pasados += 1
            if prev_red:
                if on_blue:
                    if next_checkpoint != 0:
                        next_checkpoint = self.last_checkpoint
                    else:
                        next_checkpoint -= 1
                    checkpoints_pasados -= 1
            prev_blue = on_blue
            prev_red = on_red
            solution.append(selected_coord), next_checkpoint
            step += 1
        return np.array(solution), checkpoints_pasados
            
    def _evaluate(self, solution, checkpoints_pasados) -> float:
        min_dist = min_distance_to_checkpoint(solution[-1], self.red_checkpoints[checkpoints_pasados%self.last_checkpoint+1])
        if min_dist != 0:
            return 10*checkpoints_pasados+1/min_dist
        else:
            return 10*checkpoints_pasados+2

    def _heuristic(self, candidate, next_checkpoint) -> np.ndarray:
        min_dist = min_distance_to_checkpoint(candidate, self.red_checkpoints[next_checkpoint])
        if min_dist != 0:
            return 1/min_dist
        else:
            return 2

    def _get_candidates(self, coord_actual):
        movimientos = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        candidatos = []
        for movimiento in movimientos:
            coord_siguiente = (coord_actual[0]+movimiento[0], coord_actual[1]+movimiento[1])
            if on_map(self.mapa, coord_siguiente):
                candidatos.append(coord_siguiente)
        return candidatos

    def _update_pheromone(self, trails, best_fitness):
        evaporation = 1 - self.rho
        self.pheromone *= evaporation
        for solution, fitness in trails:
            delta_fitness = 1.0/(1.0 + (best_fitness - fitness) / best_fitness)
            for coord in solution:
                self.pheromone[coord[1]][coord[0]] += delta_fitness

mapversion = 3
if mapversion == 1:
    map_path = "rsc/tomeu_map/map.png"
    blue_checkpoints_path = "rsc/tomeu_map/blue_checkpoints.png"
    red_checkpoints_path = "rsc/tomeu_map/red_checkpoints.png"
elif mapversion == 2:
    map_path = "rsc/tomeu_map_v2/map.png"
    blue_checkpoints_path = "rsc/tomeu_map_v2/blue_checkpoints.png"
    red_checkpoints_path = "rsc/tomeu_map_v2/red_checkpoints.png"
elif mapversion == 3:
    map_path = "rsc/tomeu_map_v3/map.png"
    blue_checkpoints_path = "rsc/tomeu_map_v3/blue_checkpoints.png"
    red_checkpoints_path = "rsc/tomeu_map_v3/red_checkpoints.png"

aco = ACO(map_path, blue_checkpoints_path, red_checkpoints_path, 1000, 1, 1, 20, 0.05)

solucion = aco.optimize(300)

print(solucion)




