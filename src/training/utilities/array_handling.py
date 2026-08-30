def flatten_list(two_dimensional_list: list):
    if not isinstance(two_dimensional_list, list):
        print("Error while flattening list, provided parameter is not a list.")
        quit()


    try:
        return [item for l in two_dimensional_list for item in l]
    except TypeError:
        print("Error while flattening list, list contains non-list items.\n"
              "Provided list needs to be a 2D list.")
        quit()
