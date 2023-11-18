import operator
import random
import numpy as np
from deap import base, creator, tools, gp

# Define the problem: A simple control task (e.g., setpoint tracking)
def control_system(inputs):
    # This is a simple example; you would replace this with your control logic
    # Here, let's assume a simple proportional control: output = K * error
    K = inputs[0]
    error = inputs[1]
    return K * error

# Define the genetic programming functions and terminals
pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)
pset.addTerminal(1.0)
pset.addTerminal(0.0)
pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Define the evaluation function
def evaluate(individual):
    # Compile the GP individual into a callable function
    control_function = toolbox.compile(expr=individual)

    # Simulate the control system and calculate a fitness value (e.g., tracking error)
    setpoint = np.sin(np.linspace(0, 10, 100))  # Example setpoint signal
    measured_output = np.zeros_like(setpoint)

    for i in range(len(setpoint)):
        error = setpoint[i] - measured_output[i]
        control_input = control_function([1.0, error])
        measured_output[i] = control_system([control_input, error])

    fitness = np.sum(np.abs(setpoint - measured_output))
    return fitness,

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

if __name__ == "__main__":
    # Create an initial population
    population = toolbox.population(n=100)
    generations = 20

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Evolve the population for a number of generations
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

    # Get the best individual
    best_ind = tools.selBest(population, k=1)[0]
    print("Best individual:", best_ind)
    print("Best fitness:", best_ind.fitness.values[0])
    print("Control function:", toolbox.compile(expr=best_ind))
