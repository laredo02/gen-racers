from inspyred import benchmarks,ec

import utils
track = "/rsc/track.png"
map = "/rsc/map.png"

class Genetic(benchmarks.Benchmark):
    '''
                            012
                            7C3
                            654
    '''

    def __init__(self):
        # map[y][x] : 0,0 arriba a la izquierda
        self.map = utils.image_to_matrix(map)
        # checkpoint[[(x,y)]]
        self.checkpoint = utils.extract_checkpoints(map)
        # array de movimientos
        self.movement = [(-1,-1),(0,1),(1,-1),(1,0),(1,1),(0,-1),(-1,-1),(-1,0)]
        self.origin = (x, y)


    def simulate(self, genome):

