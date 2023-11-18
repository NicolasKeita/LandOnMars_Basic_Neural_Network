import operator
import random
import numpy as np
from deap import base, creator, tools, gp, algorithms

# Define the symbolic regression problem
x = np.linspace(-1, 1, 100)
y = x**2 + 0.3 * x + 0.5 + np.random.normal(0, 0.1, 100)

# Define the primitives (functions and terminals)
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(np.square, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)
pset.addTerminal(1.0)
pset.addTerminal(-1.0)
pset.addTerminal(0.0)
pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

# Define the types for the individuals
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Define the genetic operators
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Define the evaluation function
def evaluate(individual, x, y):
    func = toolbox.compile(expr=individual)
    y_pred = [func(xi) for xi in x]
    mse = np.mean((y_pred - y)**2)
    return mse,

toolbox.register("evaluate", evaluate, x=x, y=y)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

# Main evolution loop
def main():
    population = toolbox.population(n=300)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaMuPlusLambda(population, toolbox, mu=300, lambda_=100, cxpb=0.7, mutpb=0.3, ngen=40, stats=stats, halloffame=hall_of_fame)

    # Print the best individual found
    best_individual = hall_of_fame[0]
    best_func = toolbox.compile(expr=best_individual)
    print(f"Best Individual: {best_individual}")
    # print(f"Best Expression: {gp.stringify(best_individual, pset)}")
    print(f"Fitness: {best_individual.fitness.values[0]}")

    # Plot the true vs predicted values
    import matplotlib.pyplot as plt
    plt.scatter(x, y, label='True Data')
    plt.plot(x, [best_func(xi) for xi in x], color='red', label='Predicted')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
