from random import Random
from time import time

from inspyred import benchmarks,ec

import utils
track = "rsc/demo_map/track.png"
map = "rsc/demo_map/map.png"
blue_chekpoint_path = "rsc/tomeu_map/blue_checkpoints.png"
red_chekpoint_path = "rsc/tomeu_map/red_checkpoints.png"

genome_size = 400

class Genetic(benchmarks.Benchmark):
    '''
                            012
                            7C3
                            654
    '''

    def __init__(self):
        benchmarks.Benchmark.__init__(self, genome_size)
        self.bounder = ec.DiscreteBounder([0, 1])
        self.maximize = True
        # map[y][x] : 0,0 arriba a la izquierda
        self.map = utils.image_to_matrix(map)
        # checkpoint[[(x,y)]]
        self.checkpoint_blue = utils.extract_blue_checkpoints(blue_chekpoint_path)
        self.checkpoint_red = utils.extract_red_checkpoints(red_chekpoint_path)
        # array de movimientos
        self.movement = [[-1,-1],[0,1],[1,-1],[1,0],[1,1],[0,-1],[-1,-1],[-1,0]]
        self.default_origin = utils.checkpoint_centre(self.checkpoint_red[len(self.checkpoint_red)-1])
        self.generacion = 0


    def simulate(self, genome):
        position = self.default_origin
        checkpoint = 0
        crashes = 0
        prev_blue = False
        prev_red = False
        on_blue = False
        on_red = False
        for gene in genome:
            position[0] = position[0] + self.movement[gene][0]
            position[1] = position[1] + self.movement[gene][1]
            on_map = utils.on_map(self.map, position)
            if on_map:
                on_blue = utils.on_checkpoint(self.checkpoint_blue[0], position)
                on_red = utils.on_checkpoint(self.checkpoint_red[0], position)
                if (prev_blue):
                    if (on_red):
                        checkpoint += 1
                if (prev_red):
                    if (on_blue):
                        checkpoint -= 1
                prev_blue = on_blue
                prev_red = on_red
            else:
                crashes+=1
                position = utils.checkpoint_centre(self.checkpoint_red(checkpoint-1))
        return checkpoint*30 + utils.min_distance_to_checkpoint(checkpoint-1) - 5*crashes

    def generator(self, random, args):
        """Return a candidate solution for an evolutionary algorithm."""
        return [random.choice([0, 7]) for _ in range(genome_size)]

    def evaluator(self, candidates, args):
        individuos = 0
        fitness = []
        print("generacion")
        print(self.generacion)
        self.generacion+=1
        for candidate in candidates:
            print("generacion")
            print(self.generacion)
            print(individuos)
            cand_fitness = self.simulate(candidate)
            fitness.append(cand_fitness)


problem = Genetic()

seed = time() # the current timestamp
prng = Random()
prng.seed(seed)

ga = ec.GA(prng)
ga.selector = ec.selectors.fitness_proportionate_selection
ga.variator = [ec.variators.n_point_crossover,
               ec.variators.bit_flip_mutation]
ga.replacer = ec.replacers.generational_replacement
ga.terminator = ec.terminators.generation_termination
final_pop = ga.evolve(generator=problem.generator,
                      evaluator=problem.evaluator,
                      bounder=problem.bounder,
                      maximize=problem.maximize,
                      pop_size=100,
                      max_generations=100,
                      num_elites=1,
                      num_selected=100,
                      crossover_rate=1,
                      num_crossover_points=1,
                      mutation_rate=0.05)

best = max(ga.population)
print('Best Solution: {0}: {1}'.format(str(best.candidate), best.fitness))