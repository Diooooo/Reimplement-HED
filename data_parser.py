import os
import cv2
import numpy as np
import math


def read_file_lst(file_path):
    f = open(file_path)
    file_names = f.readlines()
    f.close()

    file_names = [f_name.strip() for f_name in file_names]  # remove space and tab

    return file_names


def split_pair_names(file_names, base_dir):
    file_names = [c.split(' ') for c in file_names]
    file_names = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1])) for c in file_names]

    return file_names


def randomize(n):
    return np.random.permutation(n)


class DataParser:
    def __init__(self, batch_size, train_split=0.8, input_size=(480, 480)):
        self.train_file = './dataset/HED-BSDS/train_pair.lst'
        self.train_data_dir = './dataset/HED-BSDS'
        self.train_pairs = read_file_lst(self.train_file)
        self.samples = split_pair_names(self.train_pairs, self.train_data_dir)

        self.ids = randomize(len(self.train_pairs))
        self.train_ids = self.ids[:int(train_split * self.ids.shape[0])]
        self.validate_ids = self.ids[int(train_split * self.ids.shape[0]):]

        self.batch_size = batch_size

        self.step_per_epoch = math.ceil(len(self.train_ids) / self.batch_size)

        self.img_width = input_size[0]
        self.img_height = input_size[1]

    def get_batch_data(self, batch):
        file_names = []
        images = []
        edge_maps = []

        for idx in batch:
            image = cv2.imread(self.samples[idx][0])
            image = cv2.resize(image, (480, 480))
            edge = cv2.imread(self.samples[idx][1])
            edge = cv2.cvtColor(edge, cv2.COLOR_RGB2GRAY)
            edge = cv2.resize(edge, (480, 480))

            bin_edge = np.zeros_like(edge)
            bin_edge[np.where(edge)] = 1

            file_names.append(self.samples[idx])
            images.append(image)
            edge_maps.append(bin_edge)

        return file_names, np.asarray(images), np.asarray(edge_maps)


if __name__ == "__main__":
    pass