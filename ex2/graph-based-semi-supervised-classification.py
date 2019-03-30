import numpy as np

TRAINING_DATA_PATH = '../data/toy-data-training.csv'
TESTING_DATA_PATH = '../data/toy-data-testing.csv'
DATA_TYPE = [('id', 'i8'), ('v1', 'f8'), ('v2', 'f8'), ('v3', 'f8'), ('v4', 'f8'), ('y', 'S10')]

def weigh(node1, node2):
    node1_value = np.array(list(node1)[1:-1])
    node2_value = np.array(list(node2)[1:-1])
    return 1 / (1 + np.linalg.norm(node1_value - node2_value))

train_data = np.genfromtxt(TRAINING_DATA_PATH, names=True, dtype=DATA_TYPE, delimiter=',')
test_data = np.genfromtxt(TESTING_DATA_PATH, names=True, dtype=DATA_TYPE, delimiter=',')
all_data = np.concatenate((train_data, test_data))


for x in range(19):
    print(weigh(all_data[12], all_data[x]))