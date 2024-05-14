from inspyred import benchmarks,ec

import utils
track = "rsc/track.png"
map = "rsc/map.png"
chekpoint_path = "rsc/checkpoints.png"

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
        self.checkpoint_blue = utils.extract_checkpoints(chekpoint_path)
        self.checkpoint_red = utils.extract_checkpoints(chekpoint_path)
        # array de movimientos
        self.movement = [[-1,-1],[0,1],[1,-1],[1,0],[1,1],[0,-1],[-1,-1],[-1,0]]
        self.default_origin = utils.checkpoint_centre(self.checkpoint_red[len(self.checkpoint_red)-1])


    def simulate(self, genome):
        position = self.default_origin
        checkpoint = 0
        prev_blue = False
        prev_red = False
        on_blue = False
        on_red = False
        for gene in genome:
            position[0] = position[0] + self.movement[gene][0]
            position[1] = position[1] + self.movement[gene][1]
            on_map = utils.on_map(self.map, position)
            on_blue = utils.on_checkpoint(self.checkpoint_blue[0], position)
            on_red = utils.on_checkpoint(self.checkpoint_red[0], position)
            if(prev_blue):
                if(on_red):
                    checkpoint+=1
            if(prev_red):
                if(on_blue):
                    checkpoint-=1
            prev_blue=on_blue
            prev_red=on_red
