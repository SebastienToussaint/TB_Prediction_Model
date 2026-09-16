from torchvision import datasets
from torchvision.transforms import ToTensor
from utilities.directory_handling import verify_directory
import cv2
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from math import ceil

class CustomDataset(Dataset):
    """
    Custom dataset class to be used during dataloader creation.
    Stores a list of images and their associated labels
    """
    def __init__(self, dataset):
        self.images = []
        self.labels = []

        for datapoint in dataset:
            self.images.append(datapoint[0])
            self.labels.append(datapoint[1])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index].float(), self.labels[index]


def fashion_mnist_to_images(fashionMNIST_data_path, save_directory, resize_scale: list = None,):
    """
    Creates a png image dataset of shoes and shirts from the fashionmnist dataset.
    Saves images to the save directory and adds labels to the end of the filenames.
    :param fashionMNIST_data_path:
    :param resize_scale:
    :param save_directory:
    :return:
    """
    print("Creating and saving images")
    verify_directory(save_directory, 1)
    idx = 0
    data = load_test_data(1, 100, fashionMNIST_data_path)[0]
    for (images, labels) in data:
        for i, image in enumerate(images):
            save_name = f"{save_directory}/{idx}_fashion_{labels[i]}.png"

            # Convert image to array, resize and change dtype
            image = np.asarray(image).squeeze()
            image = (image * 65535).round().astype(np.uint16)
            if resize_scale:
                image = cv2.resize(image, (resize_scale[0], resize_scale[1]), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(save_name, image)
            idx += 1


def load_test_data(partitions: int, batch_size, fashionMNIST_data_path):
    """
    Returns randomly partitioned data in the form of a list of dataloaders of a filtered 2 class version of the
    fashionMNIST dataset.
    :param partitions: Number of partitions to split the data into
    :param batch_size: Batch size to be used in dataloaders
    :param fashionMNIST_data_path: Path to the fashionMNIST dataset. If data has not yet been downloaded, it will
    be downloaded into this directory
    :return: A list of k dataloaders containing the partitioned data where k = partitions
    """

    # Get data
    print("Retrieving FashionMNIST data...")

    train_data = datasets.FashionMNIST(fashionMNIST_data_path,True, ToTensor(), None, True)

    # Extract only two classes (2->0: Pullover and 9->1: Ankle boot, from the dataset
    print("Filtering data...")
    filtered_data = []

    for item in train_data:
        if item[1] == 2:
            filtered_data.append((item[0], 0))
        elif item[1] == 9:
            filtered_data.append((item[0], 1))

    partitioned_data = _transform_raw_data_to_dataloader(filtered_data, partitions, batch_size)
    return partitioned_data


def _transform_raw_data_to_dataloader(raw_data: list, partitions: int, batch_size: int):
    """
    Only to be used within object
    """

    print("Partitioning data...")

    datapoints_per_partition = int(ceil(len(raw_data) / partitions))
    partitioned_data = []

    for i in range(partitions):  # = [[]]*partitions causes bug because all [] have same mem address
        partitioned_data.append([])

    non_full_partitions = [n for n in range(partitions)]

    # Randomly assign each datapoint to a partition
    while len(raw_data) > 0:
        random_index = random.randint(0, len(non_full_partitions) - 1)
        assigned_partition = non_full_partitions[random_index]

        partitioned_data[assigned_partition].append(raw_data.pop(0))  # Save memory by popping instead of copying

        if len(partitioned_data[assigned_partition]) == datapoints_per_partition:
            non_full_partitions.pop(random_index)

    # Transform the partitioned data into data loaders
    for i in range(len(partitioned_data)):
        partitioned_data[i] = DataLoader(CustomDataset(partitioned_data[i]), batch_size, True)

    return partitioned_data


if __name__ == "__main__":
    fashion_mnist_to_images("E:\Project\TESTING\Fashion", "E:\Project\TESTING\Fashion\images_raw")