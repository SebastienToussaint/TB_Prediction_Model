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

