# Gets the age, gender and TB state distribution of all the data, using the metadata csv files found in the target dir.
# Can select to combine all studies into one, or keep each study separate
import csv
import matplotlib.pyplot as plt
from utilities.string_manipulation import get_column_index_for_label


def get_age_counts(data_array:list, age_groups: list):
    """
    Returns the number of individuals found within each age group in a metadata array
    :param data_array: 2D-array of metadata read from csv file
    :param age_groups: List of the youngest age of an age group: ie: [0, 18, 40] will result in the age groups
    0-17, 18-39 and 40+
    :return: age_ranges, count_per_age_group
    """
    age_array_index = get_column_index_for_label(data_array[0], "Age")

    # Setup age ranges
    age_groups.append(1000)
    age_ranges = []
    for i in range(len(age_groups) - 1):
        age_ranges.append([age_groups[i], age_groups[i+1]-1])

    # Get counts
    age_group_counts = [0]*(len(age_groups)-1)

    for row in data_array[1:]:
        age_value = row[age_array_index]
        if not age_value.isnumeric():
            print(f"Non number value {age_value} found for age in data row {row}.\n"
                  f"Not counting data point in distribution.")
        else:
            age_value = int(age_value)
            for i, age_range in enumerate(age_ranges):
                if age_range[0] <= age_value <= age_range[1]:
                    age_group_counts[i] += 1

    age_groups = []

    for age_range in age_ranges:
        age_groups.append(f"{age_range[0]}-{age_range[1]}")

    age_groups[-1] = f"{age_groups[-1][:age_groups[-1].index("-")]}+"

    return age_groups, age_group_counts

def display_age_counts(age_groups, age_group_counts):
    plt.bar(age_groups, age_group_counts, color="#1a80bb")
    plt.xlabel("Age groups (years)")
    plt.ylabel("Number of individuals")
    plt.show()


if __name__ == "__main__":
    metadata = []
    with open("./../../../ProcessedData/Shenzhen/shenzhen_metadata.csv", "r") as meta_file:
        csv_reader = csv.reader(meta_file)
        for line in csv_reader:
            metadata.append(line)

    groups, counts = get_age_counts(metadata, [0, 18, 25, 35, 45, 55, 65, 80])
    display_age_counts(groups, counts)
