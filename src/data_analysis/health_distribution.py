# Gets the age, gender and TB state distribution of all the data, using the metadata csv files found in the target dir.
# Can select to combine all studies into one, or keep each study separate
import csv
import matplotlib.pyplot as plt
from utilities.string_manipulation import get_column_index_for_label

def get_health_status_counts(data_array:list):
    """
    Returns the number of individuals within each health status group
    :param data_array: 2D-array of metadata read from csv file
    :return: dictionary of status counts
    """
    health_status_index = get_column_index_for_label(data_array[0], "Status")

    # Get counts
    status_counts = {}

    for row in data_array[1:]:
        status = row[health_status_index]
        if status == "":
            print(row)

        if status not in status_counts:
            status_counts[status] = 1
        else:
            status_counts[status] += 1

    return status_counts

def display_health_status_counts(status_counts:dict):
    labels = list(status_counts.keys())
    values = [status_counts[key] for key in labels]

    plt.bar(labels, values, color="#1a80bb")
    plt.xlabel("Health Status")
    plt.ylabel("Number of individuals")
    plt.show()


if __name__ == "__main__":
    metadata = []
    with open("./../../../ProcessedData/Shenzhen/shenzhen_metadata.csv", "r") as meta_file:
        csv_reader = csv.reader(meta_file)
        for line in csv_reader:
            metadata.append(line)

    count_dict = get_health_status_counts(metadata)
    display_health_status_counts(count_dict)
