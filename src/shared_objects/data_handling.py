import os.path
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utilities.directory_handling import verify_directory
from math import ceil, floor
import cv2
import pickle
from shared_objects import tta
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    """
    Custom dataset class to be used during dataloader creation.
    Stores a list of images and their associated labels
    """
    def __init__(self, dataset):
        self.images = []
        self.labels = []
        self.filepaths = []

        for datapoint in dataset:
            self.images.append(datapoint[0])
            self.labels.append(datapoint[1])
            self.filepaths.append(datapoint[2])

    def get_filepath(self, index):
        return self.filepaths[index]

    def get_all_filepaths(self, index):
        return self.filepaths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index].float(), self.labels[index], self.filepaths[index]


class DataStructure:
    def __init__(self):
        self.native_data_path = ""
        self.dataloader_index = 0

    @staticmethod
    def encode_image(image_path):
        """
        Returns the pixel array tensor for the given image
        :param image_path: File path to image.
        :return: Image tensor
        """
        pixel_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if pixel_array is None:
            print(f"Could not load image at {image_path}.\n"
                  f"Exiting...")
            quit()

        if pixel_array.dtype == np.uint16:
            pixel_array = pixel_array.astype(np.float32)

        pixel_array = torch.from_numpy(pixel_array)

        return pixel_array

    @staticmethod
    def get_image_label_from_name(image_name: str | Path):
        """
        Returns the number between the final _ and . of a file path or name
        """
        image_name = str(image_name)
        last_underscore_index = image_name.rindex("_")
        extension_index = image_name.rindex(".")

        try:
            return int(image_name[last_underscore_index + 1:extension_index])
        except ValueError:
            print(
                f"Error when extracting label from image name. {image_name[last_underscore_index + 1:extension_index]} is"
                f"not a valid label. (Needs to be a positive integer (or 0).")
            quit()

    def create_dataset(self, images_directories: list, partitions: int, batch_size: int, save_directory: str | Path,
                       number_of_negative_images: int|str = "ALL", number_of_positive_images: int|str = "ALL",
                       negative_split_percentage: float = None):
        """
        Encodes all the images found within each image directory provided and loads them into <partitions>-dataloaders with
        the provided batch sizes. It then saves these dataloaders in pickle format to the given save directory, naming the
        pickled file index_savefile_name.
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

        for directory in images_directories:
            directory = Path(str(directory)).resolve()
            if not directory.is_dir():
                print(f"Error while creating dataset.\n"
                      f"Provided path {directory} is not a directory.")
                quit(1)
            image_paths_list += list(directory.iterdir())

        # Check for directories in image path list
        idx = 0
        found_directory_paths = []
        while idx < len(image_paths_list):
            if image_paths_list[idx].is_dir():
                found_directory_paths.append(image_paths_list.pop(idx))
            elif not str(image_paths_list[idx]).endswith(".png"):
                print(f"Non-png file found at {image_paths_list[idx]}\n."
                      f"Removing this file from the list of images.")
                image_paths_list.pop(idx)
            else:
                idx += 1

        def recursively_search_and_add_images(dir_path, imgs_list, img_extension=".png"):
            for path in dir_path.iterdir():
                if path.is_dir():
                    recursively_search_and_add_images(path, imgs_list, img_extension)
                elif str(path).endswith(img_extension):
                    imgs_list.append(path)

        if len(found_directory_paths) > 0:
            additional_image_paths = []

            for additional_path in found_directory_paths:
                recursively_search_and_add_images(additional_path, additional_image_paths)
            print(f"CAUTION: Sub-directories were found within the provided directories for dataset creation containing"
                  f" a total of {len(additional_image_paths)} images.")

            if not input(f"Would you like to add all the images found in the sub-directories to the dataset\n(CAUTION:"
                         f" This will recursively add all the images found within all found directories downstream of "
                         f"the provided directories (Total {len(additional_image_paths)} .pngs))? Y/n: ").upper() in ["YES", "Y"]:
                print("Not adding additional images. Continuing without searching sub directories...")
            else:
                image_paths_list += additional_image_paths
                print("Successfully added additional images from subdirectories to dataset.")




        # Split positively and negatively labelled images
        while len(image_paths_list) > 0:
            if self.get_image_label_from_name(image_paths_list[0]) == 0:
                negative_image_path_list.append(image_paths_list.pop(0))
            elif self.get_image_label_from_name(image_paths_list[0]) == 1:
                positive_image_path_list.append(image_paths_list.pop(0))
            else:
                print(f"None 0 or 1 label found at {image_paths_list[0]}")
                image_paths_list.pop(0)

        del image_paths_list

        # Verify parameters
        self._verify_dataset_split_params(number_of_negative_images, number_of_positive_images, negative_split_percentage)

        # Get split counts
        number_of_negative_images, number_of_positive_images = self._get_split_values(negative_image_path_list,
                                                                                      positive_image_path_list,
                                                                                      number_of_negative_images,
                                                                                      number_of_positive_images,
                                                                                      negative_split_percentage)

        # Sample all images to create subset of requested size
        selected_images = self._select_images(negative_image_path_list, positive_image_path_list,
                            number_of_negative_images, number_of_positive_images)

        del positive_image_path_list
        del negative_image_path_list

        # Create shuffled partitions from sampled data
        partitioned_data =  self._partition_selected_image_paths(partitions, selected_images)

        # Transform partitions into dataloaders and save them to the save directory
        self._create_and_save_dataloaders(partitioned_data, batch_size, save_directory)

        del partitioned_data

        print(f"Done creating dataset. Dataloaders saved in {str(Path(str(save_directory)).resolve())}")

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

        if str(number_of_negative_images).upper() == "ALL":
            number_of_negative_images = len(negative_paths)

        if str(number_of_positive_images).upper() == "ALL":
            number_of_positive_images = len(positive_paths)

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
        if number_of_positive_images > label_counts[1]:
            print(f"Error while creating dataset.\n"
                  f"Number of available positively labeled (1) images {label_counts[1]} < "
                  f"number of required positively labeled images {number_of_positive_images}")
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

    def _create_and_save_dataloaders(self, partitioned_data, batch_size, save_directory: str | Path, overwrite_data=0):

        idx = 1

        if overwrite_data:
            shutil.rmtree(Path(str(save_directory)).resolve())
            verify_directory(Path(str(save_directory)).resolve(), 2)

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
                    quit()

        for partition in partitioned_data:
            print(f"Writing dataloader {idx}")
            raw_partition_data = []
            for path in partition:
                image_array = self.encode_image(path)
                image_label = self.get_image_label_from_name(str(path))

                raw_partition_data.append((image_array, torch.tensor(int(image_label)), str(path)))

            dataloader = DataLoader(CustomDataset(raw_partition_data), batch_size, True)

            with open(str(save_directory) + "/" + f"dataloader_{idx}", "wb") as savefile:
                pickle.dump(dataloader, savefile)

            idx += 1

        del partitioned_data

    def get_next_dataloader(self, use_native_datapath: bool = True, path_to_dataloaders:str|Path = None):

        # Get and verify dataloader paths
        if use_native_datapath:
            dataloader_directory = Path(str(self.native_data_path)).resolve()
        else:
            dataloader_directory = Path(str(path_to_dataloaders)).resolve()

        verify_directory(dataloader_directory, 0)

        dataloader_paths = list(dataloader_directory.iterdir())
        dataloader_paths.sort()

        if self.dataloader_index == len(dataloader_paths):
            self.dataloader_index = 0

        data_path = dataloader_paths[self.dataloader_index]

        self.dataloader_index += 1

        with open(data_path, "rb") as file:
            dataloader = pickle.load(file)

        if dataloader is None:
            print(f"Error while getting dataloader at {dataloader_paths[self.dataloader_index]}, could not load"
                  f" object")
            quit()

        return dataloader

    def reset_dataloader_index(self):
        self.dataloader_index = 0

    def get_num_partitions(self, use_native_datapath: bool = True, path_to_dataloaders:str|Path = None):
        # Get and verify dataloader paths
        if use_native_datapath:
            dataloader_directory = Path(str(self.native_data_path)).resolve()
        else:
            dataloader_directory = Path(str(path_to_dataloaders)).resolve()

        verify_directory(dataloader_directory, 0)

        return len(list(dataloader_directory.iterdir()))

    def delete_data(self, delete_native=False, filepath=""): #TODO add native delete
        if not delete_native and not filepath:
            print("Error in delete_data(). Need to select delete_native as true or provide filepath to file for deleting.")
            quit()

        if delete_native:
            filepath = self.native_data_path

        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"{filepath} removed successfully.")
        else:
            print(f"Could not find {filepath}")

    def set_native_datapath(self, datapath: Path | str):
        datapath = Path(str(datapath)).resolve()

        if not datapath.is_dir():
            print(f"Error while setting up dataloader datapath.\n"
                  f"{datapath} needs to be a directory containing dataloaders.")
            quit(1)

        if len(list(datapath.iterdir())) == 0:
            print(f"Error while setting up dataloader datapath.\n"
                  f"{datapath} is an empty directory.")
            quit(1)

        for path in datapath.iterdir():
            if path.is_dir():
                print(f"Error while setting up dataloader datapath.\n"
                      f"{path} is a directory. All files in the datapath directory need"
                      f" to be dataloaders.")
                quit(1)
            with open(path, "rb") as file:
                try:
                    dataloader = pickle.load(file)
                    if not isinstance(dataloader, DataLoader):
                        print(f"Error while setting up dataloader datapath.\n"
                              f"{path} is not an instance of a dataloader. All files in the datapath directory need"
                              f" to be dataloaders.")
                        quit(1)
                except Exception as e :
                    print(f"Error while setting up dataloader datapath.\n"
                          f"{path} is likely not a dataloader.\n"
                          f"Thrown exception: {e}")
                    quit(1)

        verify_directory(datapath, 0)

        self.native_data_path = datapath
        self.reset_dataloader_index()

        print("Successfully set datapath.")

    def shuffle_data(self):
        print("Shuffling data...")

        num_partitions = self.get_num_partitions()

        filepaths = []

        batch_size = 0

        for i in range(num_partitions):
            partition = self.get_next_dataloader()
            filepaths.extend(partition.dataset.get_all_filepaths())
            for images, labels in partition:
                if len(images) > batch_size:
                    batch_size = len(images)

        # Create shuffled partitions from sampled data
        partitioned_data = self._partition_selected_image_paths(num_partitions, filepaths)

        # Transform partitions into dataloaders and save them to the save directory
        self._create_and_save_dataloaders(partitioned_data, batch_size, self.native_data_path, 1)

        del partitioned_data

    @staticmethod
    def augment_data(dataloader, x_shift_range: list, y_shift_range: list, rotation_range: list):
        print("Augmenting dataloader...")

        data = []
        batch_size = 0

        for batch in dataloader:
            if len(batch[0]) > batch_size:
                batch_size = len(batch[0])

            for i in range(len(batch[0])):
                data.append([batch[0][i], batch[1][i], batch[2][i]])

        del dataloader

        data.extend(tta.augment_data_from_dataloader(data, x_shift_range, y_shift_range, rotation_range))

        dataloader = DataLoader(CustomDataset(data), batch_size, True)

        return dataloader

    @staticmethod
    def observe_dataloader( dataloader):
        """
        Shows the images within the dataloader one at a time with their associated filepath and label.
        """
        for batch in dataloader:
            batch_images = batch[0]
            batch_labels = batch[1]
            batch_filepaths = batch[2]

            batch_images = batch_images.unsqueeze(1)

            for j in range(len(batch_images)):
                img = batch_images[j].squeeze()
                label = batch_labels[j]
                plt.imshow(img, cmap="gray")
                plt.title(f"Real: {label}\n"
                          f"Filepath: {batch_filepaths[j]}")
                plt.show()
                plt.close()


if __name__ == "__main__":
    d = DataStructure()

    d.create_dataset([Path("/home/sebastien/Project/PIPELINETESTING/Data/TBX_11K/")],
                     10, 30, "/home/sebastien/Project/PIPELINETESTING/Dataloaders/TBX11K/SickVsTB_BS30/")
