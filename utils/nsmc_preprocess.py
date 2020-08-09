import random as rd
import csv


rd.seed(2020)


def read_txt(filename):
    dataset = []
    with open(filename, 'r', encoding="utf-8", newline='') as f:
        for line in f.readlines()[1:]:
            split_data = line.split("\t")
            doc = split_data[1]
            label = int(split_data[2])

            dataset.append((label, doc))

    return dataset


def save_csv(filename, dataset):
    with open(filename, 'w', encoding="utf-8", newline='') as f:
        wr = csv.writer(f)
        for data in dataset:
            wr.writerow(data)


def split_train_valid(dataset, ratio=0.8):

    num_data = len(dataset)
    print("Number of dataset : {}".format(num_data))

    sorted(dataset)

    train_size = int(num_data * ratio)

    train_data = dataset[:train_size]
    dev_data = dataset[train_size:]

    print("Number of train dataset : {}".format(len(train_data)))
    print("Number of dev dataset : {}".format(len(dev_data)))

    return train_data, dev_data


def main():
    TRAIN_PATH = "data_in/nsmc/ratings_train.txt"
    TEST_PATH = "data_in/nsmc/ratings_test.txt"
    TRAIN_SAVE_PATH = "data_in/train.csv"
    DEV_SAVE_PATH = "data_in/dev.csv"
    TEST_SAVE_PATH = "data_in/test.csv"

    dataset = read_txt(TRAIN_PATH)
    train, dev = split_train_valid(dataset)
    test = read_txt(TEST_PATH)

    save_csv(TRAIN_SAVE_PATH, train)
    save_csv(DEV_SAVE_PATH, dev)
    save_csv(TEST_SAVE_PATH, test)


if __name__ == "__main__":
    main()
