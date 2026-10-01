import sys
import csv
from pathlib import Path
import cv2
import os

images_dir = sys.argv[1]
metadata_path = sys.argv[2]

metadata = []
with open(metadata_path, "r") as file:
    csv_reader = csv.reader(file)

    for line in csv_reader:
        metadata.append(line)


if len(metadata[0]) == 12 and metadata[0][11] == '':
    metadata[0][11] = 0
elif len(metadata[0]) == 11:
    metadata[0].append(0)

header = metadata.pop(0)
header[11] = int(header[11])



index = header[11]

while index < len(metadata):
    filename = metadata[index][0]
    img = cv2.imread(images_dir + "/" + filename)

    if img is None:
        print(f"Could not find image {images_dir}/{filename}")

    img = cv2.resize(img, [800, 800])

    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ans = input("Keep image: ")
    if ans.upper() in ["QUIT", "EXIT", "E", "Q"]:
        break

    if not ans in ["", " "]:
        print("Deleting image.")
        os.remove(images_dir + "/" + filename)
        metadata.pop(index)
    else:
        index += 1
        header[11] += 1


with open(metadata_path, "w", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(header)
    csv_writer.writerows(metadata)

print("Done")

