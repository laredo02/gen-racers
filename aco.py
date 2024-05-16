from copy import deepcopy

from utils import *
import numpy as np

class ACO:

    def __init__(self, map_path: str, blue_checkpoints_path: str, red_checkpoints_path: str, sim_steps: int, n_ants: int, alpha: float = 1, beta: float = 5, rho: float = 0.8):
        self.mapa = image_to_matrix(map_path)                                   # Mapa del track
        self.blue_checkpoints = extract_blue_checkpoints(blue_checkpoints_path) # Checkpoints azules
        self.red_checkpoints = extract_red_checkpoints(red_checkpoints_path)    # Checkpoints rojos
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

        print("ants: ", self.n_ants)
        print("alpha: ", self.alpha)
        print("beta: ", self.beta)
        print("rho: ", self.rho)
        print("pheromone map shape", self.pheromone.shape)
        print_checkpoints(self.blue_checkpoints)
        print_checkpoints(self.red_checkpoints)
        # print_coord_on_track(self.mapa, self.blue_checkpoints, self.red_checkpoints, self.coord_inicial)

    def optimize(self, max_evaluations: int = 1000):
        n_evaluations = 0
        while n_evaluations < max_evaluations:
            trails = []
            for _ in range(self.n_ants):
                solution, next_checkpoint = self._construct_solution()
                fitness = self._evaluate(solution, next_checkpoint)
                trails.append((solution, fitness))
                if fitness < self.best_fitness:
                    self.best_solution = solution
                    self.best_fitness = fitness
            self.pheromone_history.append(self.pheromone.copy())
            self._update_pheromone(trails, self.best_fitness)
            self.trails_history.append(deepcopy(trails))
            self.best_fitness_history.append(self.best_fitness)
            n_evaluations += 1
            print(f"Best fitness: {self.best_fitness}")
        return self.best_solution

    def _construct_solution(self):
        solution = [self.coord_inicial]
        next_checkpoint = 0
        step = 0
        prev_blue = False
        prev_red = True
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
                    next_checkpoint += 1
            if prev_red:
                if on_blue:
                    next_checkpoint -= 1
            prev_blue = on_blue
            prev_red = on_red
            solution.append(selected_coord), next_checkpoint
            step += 1
            print(selected_coord)
            print_coord_on_track(self.mapa, self.blue_checkpoints, self.red_checkpoints, selected_coord)


        return np.array(solution), next_checkpoint
            
    def _evaluate(self, solution, next_checkpoint) -> float:    # implementar
        min_dist = min_distance_to_checkpoint(solution[-1], self.red_checkpoints[next_checkpoint])
        if min_dist != 0:
            return 10*next_checkpoint+1/min_dist
        else:
            return 10*next_checkpoint+2

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

aco = ACO("rsc/tomeu_map/map.png", "rsc/tomeu_map/blue_checkpoints.png", "rsc/tomeu_map/red_checkpoints.png", 400, 1)

aco.optimize(1)




