# Extract the image from a .dcm file and save it as a png. Filename will be PID name present in DCM file.

# TODO look into making sure that extracted images are saved at max bit depth and correct colour mode
import sys
from pathlib import Path
from PIL import Image
import pydicom


target_directory = Path(sys.argv[1])
save_directory = Path(sys.argv[2])

index = 0
for file in target_directory.iterdir():
    if file.name.endswith(".dcm"):
        dcm_file = pydicom.dcmread(file)
        image = Image.fromarray(dcm_file.pixel_array)
        image.save(f"{save_directory}/{index}_{dcm_file.PatientID}.png")
        index += 1
