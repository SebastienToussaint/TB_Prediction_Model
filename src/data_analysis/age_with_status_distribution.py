import numpy as np
from utilities.string_manipulation import get_column_index_for_label
import csv
import matplotlib.pyplot as plt

def get_age_with_status_counts(data_array:list, age_groups: list):
    """
    Returns the number of individuals found within each age group in a metadata array, separated by their health status
    :param data_array: 2D-array of metadata read from csv file
    :param age_groups: List of the youngest age of an age group: ie: [0, 18, 40] will result in the age groups
    0-17, 18-39 and 40+
    :return: age_ranges, count_per_age_group
    """
    age_array_index = get_column_index_for_label(data_array[0], "Age")
    health_status_index = get_column_index_for_label(data_array[0], "Status")

    # Setup age ranges
    age_groups.append(1000)
    age_ranges = []
    for i in range(len(age_groups) - 1):
        age_ranges.append([age_groups[i], age_groups[i+1]-1])

    # Get counts
    age_group_counts_tb = [0]*(len(age_groups)-1)
    age_group_count_sick_non_tb = [0]*(len(age_groups)-1)
    age_group_counts_healthy = [0]*(len(age_groups)-1)

    for row in data_array[1:]:
        age_value = row[age_array_index]
        if not age_value.isnumeric():
            print(f"Non number value {age_value} found for age in data row {row}.\n"
                  f"Not counting data point in distribution.")
        else:
            age_value = int(age_value)
            for i, age_range in enumerate(age_ranges):
                if age_range[0] <= age_value <= age_range[1]:
                    if row[health_status_index].upper() == "TB":
                        age_group_counts_tb[i] += 1
                    elif row[health_status_index].upper() == "HEALTHY":
                        age_group_counts_healthy[i] += 1
                    else:
                        age_group_count_sick_non_tb[i] += 1

    # Setup labels for age groups
    age_groups = []
    for age_range in age_ranges:
        age_groups.append(f"{age_range[0]}-{age_range[1]}")

    age_groups[-1] = f"{age_groups[-1][:age_groups[-1].index("-")]}+"

    return age_groups, age_group_counts_healthy, age_group_count_sick_non_tb,age_group_counts_tb

def display_age_with_status_counts(age_groups, counts: list, count_labels:list, colours:list):
    # Remove labels that have no individuals
    raw_counts = []
    legend_labels = []
    for i, label_counts in enumerate(counts):
        if not sum(label_counts) == 0:
            raw_counts.append(label_counts)
            legend_labels.append(count_labels[i])

    # Calculate proportional counts
    prop_counts = raw_counts.copy()

    totals = np.zeros(len(age_groups))

    for i in range(len(prop_counts)):
        prop_counts[i] = np.array(prop_counts[i])
        totals = totals + prop_counts[i]

    for i in range(len(prop_counts)):
        prop_counts[i] = prop_counts[i]/totals

    # Setup and plot proportional bar graph
    ax = plt.subplot(111)
    c_index = 0
    cumulative_counts = [0] * len(age_groups)

    for label_counts in prop_counts:
        ax.bar(age_groups, label_counts, bottom=cumulative_counts, color=colours[c_index])
        for i, count in enumerate(label_counts):
            cumulative_counts[i] += count
        c_index += 1

    # Resize plot to make space for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # Add raw counts inside sections of bar graphs
    for i, container in enumerate(ax.containers):
        ax.bar_label(container, labels=raw_counts[i], label_type="center", color="white")

    # Axis labels etc
    plt.title("Proportion of individuals within age groups\n with given health statuses")
    plt.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Age groups (years)")
    plt.ylabel("Proportion of individuals")
    plt.show()


if __name__ == "__main__":
    metadata = []
    with open("./../../../ProcessedData/Montgomery/montgomery_metadata.csv", "r") as meta_file:
        csv_reader = csv.reader(meta_file)
        for line in csv_reader:
            metadata.append(line)

    groups, counts_healthy, counts_sick, counts_tb = get_age_with_status_counts(metadata, [0, 18, 25, 35, 45, 55, 65, 80])
    display_age_with_status_counts(groups,
                                   [counts_healthy, counts_sick, counts_tb],
                                   ["Healthy", "Sick (non-TB)", "TB"],
                                   ["b","r"])
