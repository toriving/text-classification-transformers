import random as rd
import csv
import argparse


rd.seed(2020)


def read_txt(filename):
    dataset = []
    with open(filename, 'r', encoding="utf-8", newline='') as f:
        for line in f.readlines():
            label = line[0]
            doc = line[1:].strip()

            dataset.append((label, doc))

        print("Number of data in {} : {} ".format(filename, len(dataset)))
    return dataset


def save_csv(filename, dataset):
    with open(filename, 'w', encoding="utf-8", newline='') as f:
        wr = csv.writer(f)
        for data in dataset:
            wr.writerow(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='sst2', help='sst2 or sst5')
    args = parser.parse_args()

    if args.task.lower() == 'sst2':
        TRAIN_PATH = "data_in/sst2/stsa_binary_train.txt"
        DEV_PATH = "data_in/sst2/stsa_binary_dev.txt"
        TEST_PATH = "data_in/sst2/stsa_binary_test.txt"

    elif args.task.lower() == 'sst5':
        TRAIN_PATH = "data_in/sst5/stsa_fine_train.txt"
        DEV_PATH = "data_in/sst5/stsa_fine_dev.txt"
        TEST_PATH = "data_in/sst5/stsa_fine_test.txt"

    else:
        raise Exception("task : sst2 or sst5")


    TRAIN_SAVE_PATH = "data_in/train.csv"
    DEV_SAVE_PATH = "data_in/dev.csv"
    TEST_SAVE_PATH = "data_in/test.csv"

    train = read_txt(TRAIN_PATH)
    dev = read_txt(DEV_PATH)
    test = read_txt(TEST_PATH)

    save_csv(TRAIN_SAVE_PATH, train)
    save_csv(DEV_SAVE_PATH, dev)
    save_csv(TEST_SAVE_PATH, test)


if __name__ == "__main__":
    main()
