# Apply annotations to their applicable images. Currently only applies to Shenzhen Dataset file/json structure.
# Example dataset is given under test_data (including annotations)

import json
import sys
from PIL import Image, ImageDraw
from pathlib import Path
from utilities.directory_handling import verify_directory

class ROI:
    """
    Class to store regions of interest present on a CXR.
    """
    def __init__(self, shape_type, x_coordinates, y_coordinates, radius = 0, label = ""):
        self.type = shape_type
        self.coordinates = []
        self.radius = radius
        self.label = label

        if shape_type in ["polygon", "polyline", "point"]:
            for i in range(len(x_coordinates)):
                self.coordinates.append([x_coordinates[i], y_coordinates[i]])
        else:
            self.coordinates = [x_coordinates, y_coordinates]


# Get paths (all used as Path variable for consistency)
if len(sys.argv) != 4:
    print("Error: Incorrect usage\n"
          "Usage: make apply_json_annotations ARGS='<images_directory> <annotations_directory> <save_directory>'")
    quit()

images_directory = Path(sys.argv[1])
annotations_directory = Path(sys.argv[2])
save_directory = Path(sys.argv[3])

verify_directory(images_directory, 0)
verify_directory(annotations_directory, 0)
verify_directory(save_directory, 1)


# Find names of all TB-positive individual images
tb_positive_image_names = []
for file in images_directory.iterdir():
    if file.name[file.name.rindex("_") + 1] == "1" and file.name.endswith(".png"):
        tb_positive_image_names.append(file.name)

# Get all the json annotation files associated with an image
tb_positive_image_names.sort()
json_file_paths = {}
for image_name in tb_positive_image_names:
    base_name = image_name[0:image_name.index(".")]
    paths = []
    if Path.exists(Path(annotations_directory.as_posix() + "/annotations_1/JSON_annotations/" + base_name + ".json")):
        paths.append(annotations_directory.as_posix() + "/annotations_1/JSON_annotations/" + base_name + ".json")
    if Path.exists(Path(annotations_directory.as_posix() + "/annotations_2/JSON_annotations/" + base_name + ".json")):
        paths.append(annotations_directory.as_posix() + "/annotations_2/JSON_annotations/" + base_name + ".json")

    json_file_paths[image_name] = paths

# Apply annotations to images and save
for image_name in json_file_paths:
    print("Processing image " + image_name + "...")
    img = Image.open("test_data/images/" + image_name)

    # Extract ROIs for an image
    shapes = []
    for json_file in json_file_paths[image_name]:
        print("Applying json file " + json_file + "...")
        with open(json_file, "r") as file:
            data = json.load(file)
        data = data[list(data.keys())[0]]["regions"]

        for roi in data:
            if roi["shape_attributes"]["name"] in ["polygon","polyline"]:
                shapes.append(ROI(roi["shape_attributes"]["name"],
                                  roi["shape_attributes"]["all_points_x"],
                                  roi["shape_attributes"]["all_points_y"],
                                  label=roi["region_attributes"].popitem()[0]))
            elif roi["shape_attributes"]["name"] == "point":
                shapes.append(ROI(roi["shape_attributes"]["name"],
                                  roi["shape_attributes"]["cx"],
                                  roi["shape_attributes"]["cy"],
                                  label=roi["region_attributes"].popitem()[0]))
            elif roi["shape_attributes"]["name"] == "circle":
                shapes.append(ROI(roi["shape_attributes"]["name"],
                                  roi["shape_attributes"]["cx"],
                                  roi["shape_attributes"]["cy"],
                                  roi["shape_attributes"]["r"],
                                  label=roi["region_attributes"].popitem()[0]))

    # Apply ROIs to image
    draw = ImageDraw.Draw(img)
    for shape in shapes:
        if shape.type == "polygon":
            draw.polygon(shape.coordinates, width=5)
        elif shape.type == "polyline":
            draw.line(shape.coordinates, width=5)
        elif shape.type == "point":
            draw.point(shape.coordinates)
        elif shape.type == "circle":
            draw.circle(shape.coordinates, shape.radius, width=5)
            draw.text((shape.coordinates[0], shape.coordinates[1]), shape.label, 1000, font_size=50)

    img.save(save_directory.as_posix() + "/" + image_name[0:image_name.index(".")] + "_annotated.png")
