import random as rd
import csv


rd.seed(2020)

LABEL = {
    "contradiction": 0,
    "entailment": 1,
    "neutral": 2
}


def read_txt(filename):
    dataset = []
    with open(filename, 'r', encoding="utf-8", newline='') as f:
        for line in f.readlines()[1:]:
            split_data = line.rstrip("\n").split("\t")
            doc = split_data[0] + " [SEP] " + split_data[1]
            label = int(LABEL[split_data[2]])

            dataset.append((label, doc))

        print("Number of data in {} : {} ".format(filename, len(dataset)))

    return dataset


def save_csv(filename, dataset):
    with open(filename, 'w', encoding="utf-8", newline='') as f:
        wr = csv.writer(f)
        for data in dataset:
            wr.writerow(data)


def main():
    TRAIN_PATH = "data_in/kornli/multinli.train.ko.tsv"
    DEV_PATH = "data_in/kornli/xnli.dev.ko.tsv"
    TEST_PATH = "data_in/kornli/xnli.test.ko.tsv"
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
