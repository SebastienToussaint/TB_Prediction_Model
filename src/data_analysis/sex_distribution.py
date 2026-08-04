from utilities.string_manipulation import get_column_index_for_label
import matplotlib.pyplot as plt
import csv

def get_sex_split(data_array:list):
    sex_array_index = get_column_index_for_label( data_array[0], "Sex")

    sex_values = []
    for row in data_array[1:]:
        sex_values.append(row[sex_array_index])

    groups = list(set(sex_values))

    # Get counts
    sex_counts = {}
    for group in groups:
        sex_counts[group] = sex_values.count(group)

    return sex_counts

def display_sex_split(sex_counts: dict):
    labels = list(sex_counts.keys())
    values = [sex_counts[key] for key in labels]
    colours = ["#1a80bb", "#a00000", "#b8b8b8"]

    plt.pie(values,
            startangle=90,
            colors=colours,
            autopct="%1.1f%%",
            wedgeprops={"edgecolor" : "black",
                        "linewidth" : 1,
                        "antialiased" : True})

    plt.title("Sex Distribution", )
    plt.legend(labels)
    plt.show()


if __name__ == "__main__":
    metadata = []
    with open("./../../../ProcessedData/Montgomery/montgomery_metadata.csv", "r") as meta_file:
        csv_reader = csv.reader(meta_file)
        for line in csv_reader:
            metadata.append(line)

    display_sex_split(get_sex_split(metadata))

