import os.path
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from utilities.directory_handling import verify_directory
from math import ceil
import cv2
import pickle

# Test sets
from torchvision import datasets

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.images = []
        self.labels = []

        for datapoint in dataset:
            self.images.append(datapoint[0])
            self.labels.append(datapoint[1])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def encode_image(image_path):
    pixel_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if pixel_array is None:
        print(f"Could not load image at {image_path}.\n"
              f"Exiting...")
        quit()

    pixel_array = torch.from_numpy(pixel_array).float()

    return pixel_array

def get_image_label_from_name(image_name: str):
    last_underscore_index = image_name.rindex("_")
    extension_index = image_name.rindex(".")
    return image_name[last_underscore_index+1:extension_index]

class DataStructure:
    def __init__(self):
        self.native_data_path = ""
        self.dataloaders = []

    @staticmethod
    def _transform_raw_data_to_dataloaders(raw_data: list, partitions: int, batch_size: int):
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


    def create_compressed_dataset(self, images_directories: list, partitions: int, batch_size: int, save_directory: str | Path, savefile_name: str):
        """
        Encodes all the images found within each image directory provided and loads them into <partitions>-dataloaders with
        the provided batch sizes. It then saves these dataloaders in pickle format to the given save directory, naming the
        pickled file savefile_name.
        :param images_directories: List of directories to be searched for images
        :param partitions: Number of dataloaders to be created (number of partitions data should be split into
        :param batch_size: Number of images per batch in the dataloaders
        :param save_directory: Directory .pkl file should be saved to
        :param savefile_name: Name of save file
        """

        print("Creating compressed dataset...")
        verify_directory(save_directory, 2)

        raw_data = []
        for directory in images_directories:
            directory = Path(str(directory))
            for image in directory.iterdir():
                image_array = encode_image(image)
                image_label = get_image_label_from_name(str(image))

                raw_data.append((image_array, torch.tensor(int(image_label))))

        partitioned_data = self._transform_raw_data_to_dataloaders(raw_data, partitions, batch_size)

        # Save the dataloaders to given path and update native data path
        if not savefile_name.endswith(".pkl"):
            if "." in savefile_name:
                savefile_name = savefile_name[0:savefile_name.rindex(".")]
                print("Overwriting extension of save filename with .pkl")

            savefile_name += ".pkl"

        with open(str(save_directory) + "/" + savefile_name, "wb") as savefile:
            pickle.dump(partitioned_data, savefile)

        print(f"Compressed datafile saved at {str(save_directory) + "/" + savefile_name}")

        partitioned_data.clear()

        self.native_data_path = Path(str(save_directory) + "/" + savefile_name).resolve()


    def load_dataset(self, use_native_data: bool, pickled_data_path = ""):
        """
        Loads the pickled data at the native or given datapath.
        Native path is set when using create_compressed_dataset or set_native_datapath.
        :param use_native_data: This will make the DataStructure load the data found at it's native data path
        :param pickled_data_path: Used to load data into the DataLoaders that is not at the native data path.
        :return: Data loaders with data. Updates the objects data loaders attribute
        """
        if not use_native_data and not pickled_data_path:
            print("Incorrect arguments for load dataset. Need one of use_native_data or pickled_data_path to be True")
            quit()

        if use_native_data:
            if not self.native_data_path or not Path(self.native_data_path).exists():
                print(f"Could not find native data path {self.native_data_path}.\n"
                      f"Use create_compressed_dataset or set_native_datapath to set the native datapath")
                quit()

            print(f"Loading native data at {self.native_data_path}...")
            pickled_data_path = self.native_data_path

        elif not Path(pickled_data_path).exists():
            print(f"Could not find pickled data path {pickled_data_path}.")
            quit()

        else:
            print(f"Loading custom data at {pickled_data_path}...")

        print(pickled_data_path)
        with open(pickled_data_path, "rb") as datafile:
            data_loaders = pickle.load(datafile)

        self.dataloaders = data_loaders
        return data_loaders

    def shuffle_data(self):
        """
        Shuffles the data in the dataloaders and sets the objects dataloaders to new shuffled data.
        :return: Shuffled data
        """
        if not self.dataloaders:
            print("Dataloaders are empty. Use load_dataset to populate them.")
            quit()

        print("Shuffling data...")

        raw_data = []

        num_partitions = len(self.dataloaders)
        batch_size = 1

        # Lump data together
        while len(self.dataloaders) > 0:
            dl = self.dataloaders.pop(0)
            for i, (batch_images, batch_labels) in enumerate(dl):

                if len(batch_images) > batch_size:
                    batch_size = len(batch_images)

                for image, label in zip(batch_images, batch_labels):
                    raw_data.append((image, label))

            del dl

        # Reload randomly into dataloaders
        partitioned_data = self._transform_raw_data_to_dataloaders(raw_data, num_partitions, batch_size)

        self.dataloaders = partitioned_data
        return partitioned_data


    def delete_data(self, delete_native=False, filepath=""):
        if not delete_native and not filepath:
            print("Error in delete_data(). Need to select delete_native as true or provide filepath to file for deleting.")
            quit()
        if not str(filepath).endswith(".pkl"):
            print("File at path to delete must be a .pkl file")

        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"{filepath} removed successfully.")

        else:
            print(f"Could not find {filepath}")


    def set_native_datapath(self, datapath: Path | str):
        datapath = Path(str(datapath)).resolve()

        verify_directory(datapath, 1)

        self.native_data_path = datapath






    def fashion_mnist_to_images(self, fashionMNIST_data_path, upscale_factor,save_directory):
        """
        Creates a png image dataset of shoes and shirts from the fashionmnist dataset.
        Saves images to the save directory and adds labels to the end of the filenames.
        :param fashionMNIST_data_path:
        :param upscale_factor:
        :param save_directory:
        :return:
        """
        print("Creating and saving images")
        idx = 0
        data = self.load_test_data(1, 100, fashionMNIST_data_path)[0]
        for (images, labels) in data:
            for i, image in enumerate(images):
                save_name = f"{save_directory}/{idx}_fashion_{labels[i]}.png"
                image = torch.tensor(cv2.resize(np.array(image), (300, 300), interpolation=cv2.INTER_CUBIC))
                cv2.imwrite(save_name, (image.squeeze().numpy() * 65535).astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                idx += 1


    def load_test_data(self, partitions: int, batch_size, fashionMNIST_data_path):
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

        verify_directory(fashionMNIST_data_path, 1)

        train_data = datasets.FashionMNIST(fashionMNIST_data_path,True, ToTensor(), None,
True)

        # Extract only two classes (2->0: Pullover and 9->1: Ankle boot, from the dataset
        print("Filtering data...")
        filtered_data = []

        for item in train_data:
            if item[1] == 2:
                filtered_data.append((item[0], 0))
            elif item[1] == 9:
                filtered_data.append((item[0], 1))

        partitioned_data = self._transform_raw_data_to_dataloaders(filtered_data, partitions, batch_size)
        self.dataloaders = partitioned_data
        return partitioned_data




if __name__ == "__main__":
    d = DataStructure()


    d.create_compressed_dataset(["test_data/fashion_images/"], 10, 32, "test_data/pickled_data/", "fashion_upscaled_123")
    # d.set_native_datapath("test_data/pickled_data/fashion.pkl")
    # d.load_dataset(True)
    # d.shuffle_data()

    pass
