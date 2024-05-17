from random import Random
from time import time, sleep
import pygame
from inspyred import benchmarks, ec
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

import utils

track = "rsc/demo_map/track.png"
map = "rsc/tomeu_map/map.png"
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
        self.movement = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]]
        self.default_origin = utils.checkpoint_centre(self.checkpoint_red[len(self.checkpoint_red) - 1])
        self.generacion = 0
        self.numCheckpoints = len(self.checkpoint_red)
        pygame.init()
        self.screen = pygame.display.set_mode((800,800))
        self.track_image = pygame.transform.scale(pygame.image.load(map),(800,800))
        self.min = []
        self.max = []
        self.avg = []

    def simulate(self, genome, show):
        pygame.display.init()
        position = [self.default_origin[0], self.default_origin[1]]
        checkpoint = 0
        crashes = 0
        prev_blue = False
        prev_red = True
        on_map = utils.on_map(self.map, position)
        on_blue = utils.on_checkpoint(self.checkpoint_blue[checkpoint], position)
        on_red = utils.on_checkpoint(self.checkpoint_red[checkpoint], position)
        if show:
            self.show_map(position, [on_map, on_red, on_blue, prev_red, prev_blue])
        next_checkpoint = self.checkpoint_blue[checkpoint]
        previous_checkpoint = self.checkpoint_red[(checkpoint-1)%self.numCheckpoints]
        for gene in genome:
            position[0] = position[0] + self.movement[gene][0]
            position[1] = position[1] + self.movement[gene][1]
            on_map = utils.on_map(self.map, position)
            on_blue = utils.on_checkpoint(self.checkpoint_blue[checkpoint], position)
            on_red = utils.on_checkpoint(self.checkpoint_red[checkpoint], position)
            if show:
                self.show_map(position, [on_map, on_red, on_blue, prev_red, prev_blue])
            if on_map:
                if prev_blue:
                    if on_red:
                        checkpoint += 1
                if prev_red:
                    if on_blue:
                        checkpoint -= 1
                prev_blue = on_blue
                prev_red = on_red
                next_checkpoint = self.checkpoint_blue[checkpoint]
                previous_checkpoint = self.checkpoint_red[(checkpoint - 1) % self.numCheckpoints]
            else:
                crashes += 1
                new_postion = utils.checkpoint_centre(previous_checkpoint)
                position = [new_postion[0], new_postion[1]]
        distance_previous_checkpoint = utils.min_distance_to_checkpoint(position, previous_checkpoint)
        if checkpoint > 0:
            return checkpoint * 10 + 10 * distance_previous_checkpoint
        else:
            return -10*checkpoint

    def generator(self, random, args):
        """Return a candidate solution for an evolutionary algorithm."""
        return [random.choice([0, 1, 2, 3, 4, 5, 6, 7]) for _ in range(genome_size)]

    def evaluator(self, candidates, args):
        max_fitness = 0
        min_fitness = 1e10
        sum_fitness = 0
        fitness = []
        print("generacion ", self.generacion)
        self.generacion += 1
        for candidate in candidates:
            cand_fitness = self.simulate(candidate, False)
            if cand_fitness > max_fitness:
                max_fitness = cand_fitness
            if cand_fitness < min_fitness:
                min_fitness = cand_fitness
            sum_fitness += cand_fitness
            fitness.append(cand_fitness)
        self.min.append(min_fitness)
        self.max.append(max_fitness)
        self.avg.append(sum_fitness/100)
        return fitness

    def show_map(self, position, booleans):
        self.screen.blit(self.track_image, (0,0))
        for i in range(8):
            for j in range(8):
                self.screen.set_at((8*position[0]+i, 8*position[1]+j), (0,255,0))
        self.print_text(booleans)
        pygame.display.flip()
        sleep(0.01)

    def get_colours(self, boleans):
        colours = []
        for b in boleans:
            if b:
                colours.append((0,255,10))
            else:
                colours.append((255, 0, 0))
        return colours

    def print_text(self, booleans):
        string = ["on map", "on red", "on blue", "prev red", "prev blue"]
        coordenadas = [(700, 10), (700, 30), (700, 50), (700, 70), (700, 90)]
        colours = self.get_colours(booleans)
        pygame.font.init()
        arial = pygame.font.SysFont("Arial", 20)
        superficies = []
        for i in range(len(string)):
            self.screen.blit(arial.render(string[i], False, colours[i]), coordenadas[i])

    def show_stat(self):
        plt.scatter(range(len(self.min)), self.min, color='red', label='Min Fitness')
        plt.scatter(range(len(self.max)), self.max, color='green', label='Max Fitness')
        plt.scatter(range(len(self.avg)), self.avg, color='blue', label='Avg Fitness')
        plt.xlabel('Generacion')
        plt.ylabel('Fitness')
        plt.title('Fitness minimo, maximo, y medio en cada generación.')
        plt.legend()
        plt.show()



problem = Genetic()

seed = time()  # the current timestamp
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
                      max_generations=10000,
                      num_elites=10,
                      num_selected=100,
                      crossover_rate=1,
                      num_crossover_points=40,
                      mutation_rate=0.05)

best = max(ga.population)
print('Best Solution: {0}: {1}'.format(str(best.candidate), best.fitness))
problem.show_stat()
problem.simulate(best.candidate, True)