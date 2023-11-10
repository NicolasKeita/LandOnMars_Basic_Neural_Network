# Preprocess Data
import random
import pandas as pd
from src.ANN import ANN
from src.genetic_ann import crossover




df = pd.read_table('./diabetes.txt', header=None, encoding='gb2312', sep='\t')
df.astype(float)
# remove redundant col which is the opposite value of the 10th col
df.pop(10)
# remove first col of bias = 1
df.pop(0)
# the label column
label = df.pop(9)
# train feature
df = df.fillna(1)
train_feature = df[:576]
# train_feature = train_feature.fillna(1)
# train label
train_label = label[:576]
# test feature
test_feature = df[576:]
# test label
test_label = label[576:]

# store all active ANNs
networks = []
pool = []
# Generation counter
generation = 0
# Initial Population
population = 10
for i in range(population):
    networks.append(ANN())
# Track Max Fitness
max_fitness = 0
# Store Max Fitness Weights
optimal_weights = []

epochs = 10
# Evolution Loop
for i in range(epochs):
    generation += 1
    logging.debug("Generation: " + str(generation) + "\r\n")

    for ann in networks:
        # Propagate to calculate fitness score
        ann.forward_propagation(train_feature, train_label)
        # Add to pool after calculating fitness
        pool.append(ann)

    # Clear for propagation of next children
    networks.clear()

    # Sort anns by fitness
    pool = sorted(pool, key=lambda x: x.fitness)
    pool.reverse()

    # Find Max Fitness and Log Associated Weights
    for i in range(len(pool)):
        if pool[i].fitness > max_fitness:
            max_fitness = pool[i].fitness

            # Iterate through layers, get weights, and append to optimal
            optimal_weights = []
            for layer in pool[i].layers:
                optimal_weights.append(layer.get_weights()[0])
    # Crossover: top 5 randomly select 2 partners
    for i in range(5):
        for j in range(2):
            # Create a child and add to networks
            temp = crossover(pool[i], random.choice(pool))
            # Add to networks to calculate fitness score next iteration
            networks.append(temp)

# Create a Genetic Neural Network with optimal initial weights
ann = ANN(optimal_weights)
predict_label = ann.predict(test_feature.values)
print('Test Accuracy: %.2f' % accuracy_score(test_label, predict_label.round()))