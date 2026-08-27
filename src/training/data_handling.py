import os.path
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from h5py.h5pl import append
from matplotlib.pyplot import imshow
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from utilities.directory_handling import verify_directory
from math import ceil, floor
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
        return self.images[idx].float(), self.labels[idx]


def encode_image(image_path):
    pixel_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if pixel_array is None:
        print(f"Could not load image at {image_path}.\n"
              f"Exiting...")
        quit()

    pixel_array = torch.from_numpy(pixel_array)

    return pixel_array

def get_image_label_from_name(image_name: str|Path):
    """
    Returns the number between the final _ and . of a file path or name
    :param image_name:
    :return:
    """
    image_name = str(image_name)
    last_underscore_index = image_name.rindex("_")
    extension_index = image_name.rindex(".")

    try:
        return int(image_name[last_underscore_index+1:extension_index])
    except ValueError:
        print(f"Error when extracting label from image name. {image_name[last_underscore_index+1:extension_index]} is"
              f"not a valid label. (Needs to be a positive integer (or 0).")
        quit()

class DataStructure:
    def __init__(self):
        self.native_data_path = ""
        self.dataloaders = []

    @staticmethod
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
            for image in directory.resolve().iterdir():
                image_array = encode_image(image)
                image_label = get_image_label_from_name(str(image))

                raw_data.append((image_array, torch.tensor(int(image_label))))

        partitioned_data = self._transform_raw_data_to_dataloader(raw_data, partitions, batch_size)

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

    def create_dataset(self, images_directories: list, partitions: int, batch_size: int, save_directory: str | Path,
                       number_of_negative_images: int|str = "ALL", number_of_positive_images: int|str = "ALL",
                       negative_split_percentage: float = None):
        """
        Encodes all the images found within each image directory provided and loads them into <partitions>-dataloaders with
        the provided batch sizes. It then saves these dataloaders in pickle format to the given save directory, naming the
        pickled file savefile_name.
        :param images_directories: List of directories to be searched for images
        :param partitions: Number of dataloaders to be created (number of partitions data should be split into
        :param batch_size: Number of images per batch in the dataloaders
        :param save_directory: Directory .pkl file should be saved to
        :param number_of_negative_images: Number of negative images to put into dataset (Set to "ALL" by default)
        :param number_of_positive_images: Number of negative images to put into dataset (Set to "ALL" by default)
        :param negative_split_percentage: Will disregard number_of_negative_images and number_of_positive_images and
        will use the maximum number of images that fulfill the split percentage.
        """

        print("Creating dataset...")
        verify_directory(save_directory, 2)

        # Get list of image filenames
        image_paths_list = []
        positive_image_path_list = []
        negative_image_path_list = []

        for directory in images_directories: # Make this recursive
            directory = Path(str(directory))
            image_paths_list += list(directory.iterdir())

        while len(image_paths_list) > 0:
            if get_image_label_from_name(image_paths_list[0]) == 0:
                negative_image_path_list.append(image_paths_list.pop(0))
            elif get_image_label_from_name(image_paths_list[0]) == 1:
                positive_image_path_list.append(image_paths_list.pop(0))
            else:
                print(f"None 0 or 1 label found at {image_paths_list[0]}")
                image_paths_list.pop(0)

        del image_paths_list

        # Verify parameters and get split counts
        self._verify_dataset_split_params(number_of_negative_images, number_of_positive_images, negative_split_percentage)
        number_of_negative_images, number_of_positive_images = self._get_split_values(negative_image_path_list,
                                                                                      positive_image_path_list,
                                                                                      number_of_negative_images,
                                                                                      number_of_positive_images,
                                                                                      negative_split_percentage)

        selected_images = self._select_images(negative_image_path_list, positive_image_path_list,
                            number_of_negative_images, number_of_positive_images)

        del positive_image_path_list
        del negative_image_path_list

        partitioned_data =  self._partition_selected_image_paths(partitions, selected_images)

        self._create_and_save_dataloaders(partitioned_data, batch_size, save_directory)

        del partitioned_data


        print(f"Done creating dataset. Dataloaders saved in {str(Path(str(save_directory)).resolve())}")










        # # Save the dataloaders to given path and update native data path
        # if not savefile_base_name.endswith(".pkl"):
        #     if "." in savefile_base_name:
        #         savefile_name = savefile_base_name[0:savefile_base_name.rindex(".")]
        #         print("Overwriting extension of save filename with .pkl")
        #
        #     savefile_base_name += ".pkl"
        #
        # with open(str(save_directory) + "/" + savefile_base_name, "wb") as savefile:
        #     pickle.dump(partitioned_data, savefile)
        #
        # print(f"Compressed datafile saved at {str(save_directory) + "/" + savefile_base_name}")
        #
        # partitioned_data.clear()
        #
        # self.native_data_path = Path(str(save_directory) + "/" + savefile_base_name).resolve()

    @staticmethod
    def _verify_dataset_split_params(number_of_negative_images, number_of_positive_images, negative_split_percentage):
        """
        ALl error messages for incorrect parameter usage
        """

        if str(number_of_negative_images).upper() != "ALL" and \
            not (isinstance(number_of_negative_images, int) and number_of_negative_images >= 0):
            print(f"Error while creating dataset.\n"
                  f"{number_of_negative_images} is not a valid parameter value for number_of_negative_images. "
                  f"Must be either 'ALL' or a integer > 0")
            quit()

        if str(number_of_positive_images).upper() != "ALL" and \
            not (isinstance(number_of_positive_images, int) and number_of_positive_images >= 0):
            print(f"Error while creating dataset.\n"
                  f"{number_of_positive_images} is not a valid parameter value for number_of_positive_images. "
                  f"Must be either 'ALL' or a integer > 0")
            quit()

        # Verify class split
        if negative_split_percentage or negative_split_percentage == 0.0:
            try:
                float(negative_split_percentage)
                if not 0 <= negative_split_percentage <= 1:
                    raise ValueError

            except ValueError:
                print(f"Error while creating dataset.\n"
                      f"{negative_split_percentage} is not a valid parameter value for negative_split_percentage. "
                      f"Must be a value between 0 and 1 or None (default)")
                quit()



    @staticmethod
    def _get_split_values(negative_paths: list, positive_paths: list, number_of_negative_images, number_of_positive_images, negative_split_percentage):



        label_counts = [len(negative_paths), len(positive_paths)]

        if negative_split_percentage:
            if negative_split_percentage == 0.5:
                number_of_negative_images = min(label_counts[0], label_counts[1])
                number_of_positive_images = min(label_counts[0], label_counts[1])
            else:
                number_of_negative_images = min(label_counts[0], floor((label_counts[1]/(1-negative_split_percentage))*negative_split_percentage))
                number_of_positive_images = floor((number_of_negative_images/negative_split_percentage)*(1-negative_split_percentage))

            print(f"Parameter <negative_split_percentage> set to {negative_split_percentage}. Overriding number of negative and"
                  f"positive images to use and using maximum number of images to create split.")

        if number_of_negative_images > label_counts[0]:
            print(f"Error while creating dataset.\n"
                  f"Number of available negatively labeled (0) images {label_counts[0]} < "
                  f"number of required negatively labeled images {number_of_negative_images}")
            quit()
        if number_of_positive_images > label_counts[0]:
            print(f"Error while creating dataset.\n"
                  f"Number of available positively labeled (1) images {label_counts[1]} < "
                  f"number of required negatively labeled images {number_of_positive_images}")
            quit()

        return number_of_negative_images, number_of_positive_images

    @staticmethod
    def _select_images(negative_paths: list, positive_paths: list, number_of_negative_images, number_of_positive_images):

        print(f"\nDataset is being created with {number_of_negative_images}/{len(negative_paths)} negatively labeled images "
            f"and {number_of_positive_images}/{len(positive_paths)} positively labeled images.")

        selected_negative_images = []
        selected_positive_images = []

        while len(selected_negative_images) < number_of_negative_images:
            random_idx = random.randint(0, len(negative_paths)-1)
            selected_negative_images.append(negative_paths.pop(random_idx))

        while len(selected_positive_images) < number_of_positive_images:
            random_idx = random.randint(0, len(positive_paths)-1)
            selected_positive_images.append(positive_paths.pop(random_idx))

        combined_images = []
        combined_images += selected_negative_images
        del selected_negative_images
        combined_images += selected_positive_images
        del selected_positive_images


        return combined_images

    @staticmethod
    def _partition_selected_image_paths(num_partitions, selected_images):

        if num_partitions > len(selected_images):
            print(f"Error while creating dataset. More partitions ({num_partitions}) than images ({len(selected_images)}")

        images_per_partition = ceil(len(selected_images)/num_partitions)

        partitions = []

        for i in range(num_partitions):
            partitions.append([])

        non_full_partitions = [i for i in range(num_partitions)]

        while len(selected_images) > 0:
            rand_idx = random.randint(0, len(non_full_partitions)-1)
            selected_partition = non_full_partitions[rand_idx]

            # Make sure each partition has at least one image
            if len(partitions[selected_partition]) >= 1:
                for i, partition in enumerate(partitions):
                    if len(partition) == 0:
                        selected_partition = i

            partitions[selected_partition].append(selected_images.pop(0))

            if len(partitions[selected_partition]) == images_per_partition:
                non_full_partitions.remove(selected_partition)

        return partitions

    def _create_and_save_dataloaders(self, partitioned_data, batch_size, save_directory):

        idx = 1

        if len(list(Path(str(save_directory)).iterdir())) > 0:
            print(f"Dataloader save directory {save_directory} is not empty.")
            answer = input("Would you like to empty the directory before adding new dataloaders (E), add the new "
                     "dataloaders to the ones already in the directory (A) or quit (Q)?: ").upper()
            match answer.upper():
                case "E":
                    print("Resetting directory")
                    shutil.rmtree(Path(str(save_directory)).resolve())
                    verify_directory(Path(str(save_directory)).resolve(), 2)
                case "A":
                    print("Adding new dataloaders to directory")
                    idx += len(list(Path(str(save_directory)).iterdir()))
                case _:
                    print("Exiting...")



        for partition in partitioned_data:
            print(f"Writing dataloader {idx}")
            raw_partition_data = []
            for path in partition:
                image_array = encode_image(path)
                image_label = get_image_label_from_name(str(path))

                raw_partition_data.append((image_array, torch.tensor(int(image_label))))

            dataloader = DataLoader(CustomDataset(raw_partition_data), batch_size, True)

            with open(str(save_directory) + "/" + f"dataloader_{idx}", "wb") as savefile:
                pickle.dump(dataloader, savefile)

            idx += 1

        del partitioned_data








































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
        partitioned_data = self._transform_raw_data_to_dataloader(raw_data, num_partitions, batch_size)

        self.dataloaders = partitioned_data
        return partitioned_data


    def delete_data(self, delete_native=False, filepath=""): #TODO add native delete
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

        verify_directory(datapath, 0)

        self.native_data_path = datapath


    def fashion_mnist_to_images(self, fashionMNIST_data_path, save_directory, resize_scale: list = None,):
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
        data = self.load_test_data(1, 100, fashionMNIST_data_path)[0]
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

        partitioned_data = self._transform_raw_data_to_dataloader(filtered_data, partitions, batch_size)
        self.dataloaders = partitioned_data
        return partitioned_data




if __name__ == "__main__":
    d = DataStructure()
    # d.fashion_mnist_to_images(Path("test_data/fashionM"), "test_data/fashion_images_large", [512, 512])

    d.create_dataset(["test_data/fashion_images_large/"], 10, 32, "test_data/pickled_data/partitions/",
                     1000, 1000)

