
from utils import *
import numpy as np
from typing import List, Dict

MAXMOVES = 400

class Car:
    def __init__(self, coordinates):
        self.checkpoint_number = 0
        self.coordinates = coordinates
        self.prev_blue = False
        self.prev_red = False
        self.on_blue = False
        self.on_red = False

    def move_car(self, move_number): # Devuelve la coordenada donde se movera el coche en caso de aplicar el movimiento del número parametro pasado, pero no las almacena.
        last_coordinate = self.coordinates[-1]
        movements = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        return tuple(map(sum, zip(last_coordinate, movements[move_number])))
    
    def add_cordinate(self, new_coordinate): # Añade la coordenada pasada como parametro a la lista de coordenadas y actualiza si ha pasado por un checkpoint
        self.coordinates.append(new_coordinate)
        # self.on_blue = utils.on_checkpoint(self.checkpoint_blue[0], new_coordinate)
        # self.on_red = utils.on_checkpoint(self.checkpoint_red[0], new_coordinate)
        if(self.prev_blue):
            if(self.on_red):
                self.checkpoint_number+=1
        if(self.prev_red):
            if(self.on_blue):
                self.checkpoint_number-=1
        self.prev_blue=self.on_blue
        self.prev_red=self.on_red
    
    def get_position(self): # Devuelve la posicion actual
        return self.coordinates[-1]


class ACO:

    def __init__(self, map_path: str, blue_checkpoints_path: str, red_checkpoints_path: str, n_ants: int, alpha: float = 1, beta: float = 5, rho: float = 0.8):
        self.mapa = image_to_matrix(map_path)
        self.blue_checkpoints = extract_blue_checkpoints(blue_checkpoints_path)
        self.red_checkpoints = extract_red_checkpoints(red_checkpoints_path)
        self.coord_inicial = checkpoint_middle_pixel(self.red_checkpoints[len(self.red_checkpoints)-1])

        self.n_ants = n_ants    # N de hormigas por iteracion
        self.alpha = alpha      # Influencia de la feromona
        self.beta = beta        # Influencia de la heuristica
        self.rho = rho          # Evaporacion

        self.pheromone_history = []
        self.trails_history = []
        self.best_fitness_history = []

        print("ants: ", self.n_ants)
        print("alpha: ", self.alpha)
        print("beta: ", self.beta)
        print("rho: ", self.rho)
        print_checkpoints(self.blue_checkpoints)
        print_checkpoints(self.red_checkpoints)
        print_coord_on_track(self.mapa, self.blue_checkpoints, self.red_checkpoints, self.coord_inicial)

    def optimize(self, max_evaluations: int = 1000):
        self._initialize()
        car = ([Car(self.coord_inicial) for _ in range(self.n_ants)]) # instanciar lista de coches
        n_evaluations = 0

        while n_evaluations < max_evaluations:
            trails = []
            ant = 0

            for _ in range(self.n_ants):
                solution = self._construct_solution()
                fitness = self._evaluate(solution)
                n_evaluations += 1
                trails.append((solution, fitness))

                if fitness < self.best_fitness:
                    self.best_solution = solution
                    self.best_fitness = fitness

            self.trails_history.append(deepcopy(trails))
            self.best_fitness_history.append(self.best_fitness)
            self._update_pheromone(trails)

            print(f"Best fitness: {self.best_fitness}")

        return self.best_solution

    def _initialize():
        self.pheromone = np.ones() # falta algo
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.pheromone_history = []
        self.trails_history = []
        self.best_fitness_history = []

    def _construct_solution(self) -> List[int]: 
        solution = None # Solucion inicialmente vacia

        while moves < MAXMOVES:
            candidates = self._get_candidates() # Candidatos estaticos, previa eleccion no excluyente
            pheromones = self.pheromone**self.alpha
            heuristic = self._heuristic(candidates)**self.beta
        
            total = np.sum(pheromones * heuristic)
            probabilities = (pheromones * heuristic) / total

            selected_move = np.random.choice(candidates, p=probabilities)
            solution[selected_move] = 1
            
            new_coordinate = car[ant].move_car(selected_move)
            car[ant].add_coordinate(new_coordinate)

            moves += 1

    def _evaluate(self, solution: List[int]) -> float: # implementar
        mask = np.argwhere(solution == 1).flatten() # busca los índices en la solución donde el valor es igual a 1, lo que indica que un elemento está incluido en la mochila. El resultado es un array con los índices de los elementos seleccionados.
        if np.sum(self.weights[mask]) > self.max_capacity:
            return float('-inf')
        return np.sum(self.values[mask])  

    def _heuristic(self, candidates: List[int]) -> np.ndarray:
        return 10 / utils.min_distance_to_checkpoint(car[ant].move_car(candidates) , car[ant].checkpoint_number) # heuristica de distancia minima a siguiente checkpoint 

    def _get_candidates(self):
        '''
        funcion para generar los posibles candidatos, existen 8 posiciones
        a las que se puede mover el coche la C representa el coche, los numeros son
        la posicion a la que se movera el coche.
        012
        7C3
        654
        '''
        return np.array([0, 1, 2, 3, 4, 5, 6, 7])




# map_path: str, blue_checkpoints_path: str, red_checkpoints_path: str, n_ants: int, alpha: float = 1, beta: float = 5, rho: float = 0.8):

aco = ACO("rsc/tomeu_map/map.png", "rsc/tomeu_map/blue_checkpoints.png", "rsc/tomeu_map/red_checkpoints.png", 1)

















