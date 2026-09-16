from pathlib import Path

def extract_num_from_string(s):
    num = ""
    for char in s:
        if char.isnumeric():
            num += char
    return num

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

def get_column_index_for_label(labels: list, target_label: str):
    """
    :param labels: Header row of metadata (row containing labels of columns)
    :param target_label: Label whose column number to look for
    :return: Column number of label
    """
    index = -1

    for i, col in enumerate(labels): # Done to allow for case insensitivity in index searching
        if col.upper() == target_label.upper():
            index = i

    # Could not find age column. Get user input.
    if index == -1:
        answer = input(f"Could not find labelled {target_label} column in metadata.\n"
                 "Please indicate which column number contains this data"
                       " (or type a non-integer to quit): ")

        if not answer.isnumeric():
            print("Exiting...")
            quit()
        else:
            index = int(answer)

    return index
