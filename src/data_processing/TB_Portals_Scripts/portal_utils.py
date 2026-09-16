from pathlib import Path

def extract_filename_from_path(filepath, remove_extension):

    filepath = str(filepath) # In case of Path given instead of string (might not be necessary, just for safety)
    if remove_extension:
        if "." in filepath:
            if "\\" in filepath and "/" in filepath:
                return filepath[max(filepath.rindex("\\"), filepath.rindex("/"))+1 : filepath.rindex(".")]
            elif "\\" in filepath:
                return filepath[filepath.rindex("\\") + 1 : filepath.rindex(".")]
            elif "/" in filepath:
                return filepath[filepath.rindex("/") + 1 : filepath.rindex(".")]
            else:
                return filepath[ : filepath.rindex(".")]
        else:
            if "\\" in filepath and "/" in filepath:
                return filepath[max(filepath.rindex("\\"), filepath.rindex("/"))+1:]
            elif "\\" in filepath:
                return filepath[filepath.rindex("\\") + 1:]
            elif "/" in filepath:
                return filepath[filepath.rindex("/") + 1:]
            else:
                return filepath
    else:
        if "\\" in filepath and "/" in filepath:
            return filepath[max(filepath.rindex("\\"), filepath.rindex("/")) + 1:]
        elif "\\" in filepath:
            return filepath[filepath.rindex("\\") + 1:]
        elif "/" in filepath:
            return filepath[filepath.rindex("/") + 1:]
        else:
            return filepath

def extract_base_directory_from_path(filepath):

    filepath = str(Path(str(filepath)).resolve()) # In case of Path given instead of string (might not be necessary, just for safety)

    if "\\" in filepath and "/" in filepath:
        return filepath[0: max(filepath.rindex("\\"), filepath.rindex("/")) + 1]
    elif "\\" in filepath:
        return filepath[0: filepath.rindex("\\") + 1]
    elif "/" in filepath:
        return filepath[0: filepath.rindex("/") + 1]
    else:
        return filepath

def verify_directory(directory, verification_type: int):
    if verification_type not in [0, 1]:
        print(f"Error in verify_directory(): {verification_type} is not an option.\n"
              f"Options: 0, 1")
        quit()

    if not isinstance(directory, Path):
        if isinstance(directory, str):
            directory = Path(directory)
        else:
            print(f"{directory} is not a valid name for verify_directory()")
            quit()

    if verification_type == 0:
        if not directory.exists():
            print(f"Could not find {directory.resolve()}.\nExiting...")
            quit()

    else:
        # Create save directory if it doesn't already exist
        if not directory.exists():
            if input(f"Could not find {directory.resolve()}\n"
                     f"Create a directory at this location? (Y/n): ").upper() not in ("Y", "YES"):
                print("Not creating directory\nExiting...")
                quit()
            else:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except (PermissionError, Exception):
                    print(f"Could not create {directory.absolute()}.\n"
                          f"Potentially missing permissions to create directory\n"
                          f"Exiting...")
                    quit()

                print("Successfully created directory.\nContinuing with job...")

