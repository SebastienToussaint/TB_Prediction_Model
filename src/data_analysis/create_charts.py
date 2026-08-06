import sys
from utilities.directory_handling import verify_directory
from utilities.directory_handling import get_csv_file_paths
from pathlib import Path
import csv

# Chart imports
import age_distribution
import age_with_status_distribution
import health_distribution
import sex_distribution

# Argument management and csv path retrieval
csv_paths = []
charts_to_create = []

if len(sys.argv) == 2 and sys.argv[1].upper() == "LIST":
    print(f"Available charts:\n"
          f"age\n"
          f"sex\n"
          f"status\n"
          f"age-status")
    quit()
elif len(sys.argv) >= 3:
    reached_filepaths_tag = False

    for i in range(1, len(sys.argv)):
        if sys.argv[i].upper() == "/":
            reached_filepaths_tag = True
            continue

        # Get the charts that need to be created
        if not reached_filepaths_tag:
            charts_to_create.append(sys.argv[i].upper())

        # Get the directories/filepaths for CSVs
        else:
            verify_directory(sys.argv[i], 0)

            if sys.argv[i].endswith(".csv"):
                csv_paths.append(sys.argv[i])
            else:
                csv_paths += get_csv_file_paths(Path(sys.argv[i]))
else:
    print("ERROR: Incorrect arguments provided\n"
          "Usage:\n"
          "make create_chart ARGS='list' -> will print out a list of available charts to place in chart type\n"
          "OR\n"
          "make create_chart ARGS='<[chart_types]> / <directory_to_search or csv_filepath> (optional) <[additional directories of csv paths]>'")
    quit()

if not reached_filepaths_tag:
    print("ERROR: Missing metadata filepaths marker </>\n"
          "Usage:\n"
          "make create_chart ARGS='list' -> will print out a list of available charts to place in chart type\n"
          "OR\n"
          "make create_chart ARGS='<[chart_types]> / <directory_to_search or csv_filepath> (optional) <[additional directories of csv paths]>'")
    quit()


# Get metadata
metadata = []
for path in csv_paths:
    with open(path, "r") as meta_file:
        csv_reader = csv.reader(meta_file)
        for line in csv_reader:
            if not {"AGE", "Age", "age", "SEX", "Sex", "sex", "Associated CXR", "Status", "STATUS",
                    "status"}.isdisjoint(line) and len(metadata) != 0:
                continue

            metadata.append(line)

# Call requested functions to create charts
if "SEX" in charts_to_create:
    while "SEX" in charts_to_create:
        charts_to_create.remove("SEX")
    sex_distribution.display_sex_split(sex_distribution.get_sex_split(metadata))

if "AGE" in charts_to_create:
    while "AGE" in charts_to_create:
        charts_to_create.remove("AGE")
    groups, counts = age_distribution.get_age_counts(metadata, [0, 18, 25, 35, 45, 55, 65, 80])
    age_distribution.display_age_counts(groups, counts)

if "STATUS" in charts_to_create:
    while "STATUS" in charts_to_create:
        charts_to_create.remove("STATUS")
    count_dict = health_distribution.get_health_status_counts(metadata)
    health_distribution.display_health_status_counts(count_dict)

if "AGE-STATUS" in charts_to_create:
    while "AGE-STATUS" in charts_to_create:
        charts_to_create.remove("AGE-STATUS")

    groups, counts_healthy, counts_sick, counts_tb = age_with_status_distribution.get_age_with_status_counts(
        metadata, [0, 18, 25, 35, 45, 55, 65, 80])
    age_with_status_distribution.display_age_with_status_counts(groups,
                                   [counts_healthy, counts_sick, counts_tb],
                                   ["Healthy", "Sick (non-TB)", "TB"],
                                   ["b","r"])


if len(charts_to_create) > 0:
    print("Could not create charts of following types (<make create_chart ARGS='list'> to see list of options):")
    for chart in charts_to_create:
        print(chart)

