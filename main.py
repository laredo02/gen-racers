
import numpy as np
from copy import deepcopy
from typing import List, Dict
import random

from utils import image_to_matrix, extract_checkpoints
from utils import print_map, print_checkpoints, print_track, print_coord_on_track
from utils import on_map, on_checkpoint

class GenRacers:

    def __init__(self, map_path: str, checkpoints_path: str, n_ants: int, alpha: float = 1, beta: float = 5, rho: float = 0.8):
        mapa = image_to_matrix(map_path)
        print_map(mapa)
        checkpoints = extract_checkpoints(checkpoints_path)
        print_checkpoints(checkpoints)
        print_track(mapa, checkpoints)


        print(checkpoints[0][0])
        print(on_map(mapa, checkpoints[0][0]))
        print(on_checkpoint(checkpoints[0], checkpoints[0][0] + (0, 1)))

        print_coord_on_track(mapa, checkpoints, (1, 0))


        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.pheromone_history = []
        self.trails_history = []
        self.best_fitness_history = []
        

    #def optimize():
        

    #def _initialize():

    #def _evaluate():

    #def _construct_solution():

    #def _heuristic():

    def _get_candidates(self):
        '''
        funcion para generar los posibles candidatos, existen 8 posiciones
        a las que se puede mover el coche la C representa el coche, los numeros son
        la posicion a la que se movera el coche.
        012
        7C3
        654
        '''
        return random.randint(1, 7)

    #def _update_pheromone():


genracers = GenRacers("rsc/map.png", "rsc/checkpoints.png", 3)




