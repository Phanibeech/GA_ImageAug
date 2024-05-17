import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
from pycocotools.coco import COCO
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns

# Configuration paths
dataDir = '/Users/phanindrabeechani/Desktop/coco'
dataType = 'train2017'
annFile = f'{dataDir}/annotations/instances_{dataType}.json'

# Initialize COCO api for the instances annotations
coco = COCO(annFile)
catIds = coco.getCatIds(catNms=['person', 'book', 'orange'])  # Example categories
imgIds = coco.getImgIds(catIds=catIds)

def load_and_preprocess_image(img_id):
    img_info = coco.loadImgs(img_id)[0]
    path = os.path.join(dataDir, dataType, img_info['file_name'])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))  # Reduced image size for speed
    image = image.astype(np.float32) / 255.0
    label = np.zeros((len(catIds),))
    annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for ann in anns:
        label[catIds.index(ann['category_id'])] = 1
    return image, label

def prepare_dataset():
    dataset = tf.data.Dataset.from_generator(
        lambda: (load_and_preprocess_image(img_id) for img_id in imgIds),
        output_types=(tf.float32, tf.float32),
        output_shapes=((128, 128, 3), (len(catIds),))
    )
    return dataset.repeat().batch(10).prefetch(tf.data.AUTOTUNE)

def create_model(params):
    num_filters_conv1, num_filters_conv2, num_neurons_dense, learning_rate = params
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Conv2D(int(num_filters_conv1), (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(int(num_filters_conv2), (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(int(num_neurons_dense), activation='relu'),
        Dense(len(catIds), activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Genetic Algorithm Setup
population_size = 10
num_generations = 5
mutation_rate = 0.1
crossover_rate = 0.7

# Initialize population with reduced search space
population = [
    [random.choice([16, 32, 64]), random.choice([32, 64, 128]), random.choice([64, 128, 256]), random.uniform(0.0005, 0.001)]
    for _ in range(population_size)
]

# Function to perform tournament selection
def tournament_selection(population, scores, k=3):
    selection_ix = np.random.randint(len(population))
    for ix in np.random.randint(0, len(population), k-1):
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

# Function to perform single point crossover
def crossover(p1, p2):
    if random.random() < crossover_rate:
        pt = random.randint(1, len(p1) - 2)
        p1[pt:], p2[pt:] = p2[pt:], p1[pt:]
    return [p1, p2]

# Function to perform mutation
def mutation(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            if i < 3:  # Adjust filters/neurons
                change_factor = random.choice([0.5, 2])
                individual[i] = int(individual[i] * change_factor)
                if i < 2:  # Ensure filter counts are valid
                    individual[i] = max(16, min(individual[i], 256))
            else:  # Adjust learning rate
                individual[i] *= random.choice([0.5, 1.5])
                individual[i] = max(min(individual[i], 0.001), 0.0001)  # Ensure learning rate bounds are respected
    return individual

# Training and evolution process
all_fitness_scores = []
all_params = []

for generation in range(num_generations):
    print(f'Generation {generation + 1}')
    fitness_scores = []
    generation_params = []
    for individual in population:
        print(f'Training with params: {individual}')
        model = create_model(individual)
        train_dataset = prepare_dataset()
        history = model.fit(train_dataset, epochs=1, steps_per_epoch=10, verbose=1)
        fitness = 1 / (history.history['loss'][-1] + 1e-6)
        fitness_scores.append(fitness)
        generation_params.append(individual)
        print(f'Fitness: {fitness}')
    all_fitness_scores.append(fitness_scores)
    all_params.append(generation_params)

    # Natural selection
    new_population = []
    for _ in range(int(population_size/2)):
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)
        children = crossover(parent1.copy(), parent2.copy())
        new_population.extend(children)

    # Applying mutation
    population = [mutation(ind) for ind in new_population]

best_index = np.argmax(fitness_scores)
best_params = population[best_index]
print(f'Best params: {best_params}')

# Evaluate the best model
best_model = create_model(best_params)
best_history = best_model.fit(prepare_dataset(), epochs=15, steps_per_epoch=100)

# Scatter Plot
plt.figure(figsize=(10, 5))
for i, scores in enumerate(all_fitness_scores):
    plt.scatter([i + 0.1 * random.random() for _ in scores], scores, alpha=0.6, label=f'Gen {i+1}')
plt.title('Scatter Plot of Fitness Scores Across Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness Score')
plt.legend()
plt.show()

# Line Plot
mean_fitness_scores = [np.mean(scores) for scores in all_fitness_scores]
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_generations + 1), mean_fitness_scores, marker='o', linestyle='-', color='b')
plt.title('Line Plot of Average Fitness Scores per Generation')
plt.xlabel('Generation')
plt.ylabel('Average Fitness Score')
plt.grid(True)
plt.xticks(range(1, num_generations + 1))
plt.show()

# Heatmap
param_matrix = np.zeros((num_generations, len(population)))
for i, gen in enumerate(all_params):
    for j, params in enumerate(gen):
        param_matrix[i, j] = params[-2]  # Assuming accuracy is second last in the list

sns.heatmap(param_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=[f'Ind {j+1}' for j in range(len(population))],
            yticklabels=[f'Gen {i+1}' for i in range(num_generations)])
plt.title('Heatmap of Accuracy for Different Parameter Combinations Across Generations')
plt.xlabel('Individual')
plt.ylabel('Generation')
plt.show()
