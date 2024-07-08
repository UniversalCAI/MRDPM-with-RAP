import numpy as np
import random


def generate_random_data_devide(num):
    # Generate the labels in {i:03d} format
    labels = [f"{i:03d}" for i in range(num)]

    # Shuffle the labels to randomly distribute them
    random.shuffle(labels)

    # Split the labels into 80% for training and 20% for testing
    split_index = int(0.8 * len(labels))
    train_labels = labels[:split_index]
    test_labels = labels[split_index:]

    # Save the labels into txt files
    with open('./train_list.txt', 'w') as train_file:
        train_file.write('\n'.join(train_labels))

    with open('./test_list.txt', 'w') as test_file:
        test_file.write('\n'.join(test_labels))

def generate_random_dataset(num):
    generate_random_data_devide(num)
    loc = ".\\dataset"
    for i in range(0, num):
        random_array_tmax = np.random.rand(256, 256)
        filename_tmax = loc+"\\Tmax\\"+f"{i:03d}.npy"
        np.save(filename_tmax, random_array_tmax)

        random_array_tmax = np.random.rand(256, 256)
        filename_tmax = loc+"\\mask\\"+f"{i:03d}.npy"
        np.save(filename_tmax, random_array_tmax)

        random_array_tmax = np.random.rand(256, 256, 4)
        filename_tmax = loc+"\\mCTA\\"+f"{i:03d}.npy"
        np.save(filename_tmax, random_array_tmax)
    

generate_random_dataset(20)



