# Extract the image from a .dcm file and save it as a png. Filename will be PID name present in DCM file.

# TODO look into making sure that extracted images are saved at max bit depth and correct colour mode
import sys
from pathlib import Path
from PIL import Image
import pydicom
from utilities.directory_handling import verify_directory

# Argument management
if len(sys.argv) != 3:
    print("ERROR: Incorrect arguments provided\n"
          "Usage: make extract_dcm_image ARGS='<directory_to_search> <save_directory>'")
    quit()

target_directory = Path(sys.argv[1])
save_directory = Path(sys.argv[2])

verify_directory(target_directory, 0)
verify_directory(save_directory, 1)

# Get number of dcm files in directory
num_files = sum(str(item).endswith(".dcm") for item in list(target_directory.iterdir()))

# Extract images to save directory
index = 0
for file in target_directory.iterdir():
    if file.name.endswith(".dcm"):
        print(f"Converting {file.resolve()} ({index+1}/{num_files})")
        dcm_file = pydicom.dcmread(file)
        image = Image.fromarray(dcm_file.pixel_array)
        image.save(f"{save_directory}/{index}_{dcm_file.PatientID}.png")
        index += 1

print("Done!")