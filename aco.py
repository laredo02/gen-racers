import pygame

from utils import *
import numpy as np
from typing import List, Dict

# Terminar logica de checkpoints
# Heuristica?
# Evaluate?
# Update pheromone

# GUI
# Metricas

class ACO:

    def __init__(self, map_path: str, blue_checkpoints_path: str, red_checkpoints_path: str, sim_steps: int, n_ants: int, alpha: float = 1, beta: float = 5, rho: float = 0.8):
        self.mapa = image_to_matrix(map_path)
        self.blue_checkpoints = extract_blue_checkpoints(blue_checkpoints_path)
        self.red_checkpoints = extract_red_checkpoints(red_checkpoints_path)
        self.coord_inicial = checkpoint_middle_pixel(self.red_checkpoints[len(self.red_checkpoints)-1])
        self.sim_steps = sim_steps
        self.n_ants = n_ants            # N de hormigas por iteracion
        self.alpha = alpha              # Influencia de la feromona
        self.beta = beta                # Influencia de la heuristica
        self.rho = rho                  # Evaporacion
        self.pheromone_history = []     # Matriz bidimensional de feromonas (mapa de feromonas)
        self.trails_history = []        # Lista de Listas de coordenadas (lista de caminos recorridos por hormigas)
        self.best_fitness_history = []  # Historial de bejor fitnes de cada iteracion
        self.pheromone = np.ones((len(self.mapa), len(self.mapa[0])))
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.pheromone_history = []
        self.trails_history = []
        self.best_fitness_history = []

        print("ants: ", self.n_ants)
        print("alpha: ", self.alpha)
        print("beta: ", self.beta)
        print("rho: ", self.rho)
        print("pheromone map shape", self.pheromone.shape)
        print_checkpoints(self.blue_checkpoints)
        print_checkpoints(self.red_checkpoints)
        print_coord_on_track(self.mapa, self.blue_checkpoints, self.red_checkpoints, self.coord_inicial)

    def optimize(self, max_evaluations: int = 1000):
        n_evaluations = 0
        while n_evaluations < max_evaluations:
            trails = []
            for _ in range(self.n_ants):
                solution = self._construct_solution()
                fitness = self._evaluate(solution)
                trails.append((solution, fitness))
                if fitness < self.best_fitness:
                    self.best_solution = solution
                    self.best_fitness = fitness
            n_evaluations += 1
            self.trails_history.append(deepcopy(trails))
            self.best_fitness_history.append(self.best_fitness)
            self._update_pheromone(trails)
            print(f"Best fitness: {self.best_fitness}")
        return self.best_solution

    def _construct_solution(self, coord_inicial) -> List[int]:
        solution = [(coord_inicial)]
        while moves < self.sim_steps:
            coord_actual = solution[-1]
            candidates = self._get_candidates(coord_actual) # Candidatos (coordenadas accesibles y que estan dentro del track)
            pheromones = []
            heuristic = []
            for candidate in candidates
                pheromones.append(self.pheromone[candidate[1]][candidate[0]]**self.alpha)
                heuristic.append(self._heuristic(candidates, next_checkpoint)**self.beta)
            total = np.sum(pheromones * heuristic)
            probabilities = (pheromones * heuristic) / total
            selected_coord = np.random.choice(candidates, p=probabilities)
            solution.append(selected_coord), next_checkpoint
            
    def _evaluate(self, solution: List[int]) -> float: # implementar
        mask = np.argwhere(solution == 1).flatten() # busca los índices en la solución donde el valor es igual a 1, lo que indica que un elemento está incluido en la mochila. El resultado es un array con los índices de los elementos seleccionados.
        if np.sum(self.weights[mask]) > self.max_capacity:
            return float('-inf')
        return np.sum(self.values[mask])  

    def _heuristic(self, candidates: List[int], next_checkpoint: int) -> np.ndarray:
        return 10 / utils.min_distance_to_checkpoint(car[ant].move_car(candidates) , next_checkpoint) # heuristica de distancia minima a siguiente checkpoint 

    def _get_candidates(self, coord_actual):
        movimientos = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]]
        candidatos = []
        for movimiento in movimientos()
            coord_siguiente = (coord_actual[0]+movimiento[0], coord_actual[1]+movimiento[1])
            if on_map(self.mapa, coord_siguiente)
                candidatos.append(coord_siguiente)
        return candidatos

    def _update_pheromone():
        pass
aco = ACO("rsc/tomeu_map/map.png", "rsc/tomeu_map/blue_checkpoints.png", "rsc/tomeu_map/red_checkpoints.png", 400, 1)

