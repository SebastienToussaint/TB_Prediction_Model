from torch import nn
from pathlib import Path
from utilities.directory_handling import verify_directory

def call_with_defaults(func, **kwargs):
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if value is not None
    }
    return func(**kwargs)


def parse_model_template(path):
    layer_stack = []
    with open(path, "r") as file:
        idx = 0
        for line in file:
            idx += 1
            line = line.strip()
            if line == "":
                continue
            if line[0] == "#":
                continue

            if "(" not in line or ")" not in line:
                print(f"ERROR while decoding model template at {path}\n"
                      f"Missing '(' or ')' in '{line}' provided at line {idx}.")
                quit(1)

            args = line[line.index("(")+1:line.rindex(")")]
            match line[0:line.index("(")].upper():

                # CONVOLUTIONAL LAYERS =================================================================================
                case "CONV2D":
                    if "IN_CHANNELS=" not in args.upper() or "OUT_CHANNELS=" not in args.upper() \
                            or "KERNEL_SIZE=" not in args.upper():
                        print(f"Required parameter missing for Conv2d() layer at line {idx}\n"
                              f"Provided: {line}\n"
                              f"Usage: Conv2d(in_channels=x, out_channels=y, kernel_size=z, "
                              f"(OPTIONAL: stride, padding, dilation)")
                        quit(1)

                    args = args.split(",")

                    # Initiate and setup arguments

                    in_channels, out_channels, kernel_size = None, None, None
                    stride, padding, dilation = None, None, None

                    for argument in args:
                        if argument.count("=") != 1:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Missing '=' or ','.")
                            quit(1)
                        try:
                            match argument[0:argument.index("=")].replace(" ", "").upper():
                                case "IN_CHANNELS":
                                    in_channels = int(argument[argument.index("=")+1:])
                                case "OUT_CHANNELS":
                                    out_channels = int(argument[argument.index("=") + 1:])
                                case "KERNEL_SIZE":
                                    kernel_size = int(argument[argument.index("=") + 1:])
                                case "STRIDE":
                                    stride = int(argument[argument.index("=") + 1:])
                                case "PADDING":
                                    padding = int(argument[argument.index("=") + 1:])
                                case "DILATION":
                                    dilation = int(argument[argument.index("=") + 1:])
                                case _:
                                    print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                          f"Unknown argument '{argument}' provided.\n"
                                          f"Usage: Conv2d(in_channels=x, out_channels=y, kernel_size=z,"
                                          f" (OPTIONAL: stride, padding, dilation)")
                                    quit(1)
                        except ValueError:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Incorrect value provided for argument. All values must be integers.")
                            quit(1)

                    layer_stack.append(call_with_defaults(nn.Conv2d, in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                                 dilation=dilation))

                case "CONVTRANSPOSE2D":
                    if "IN_CHANNELS=" not in args.upper() or "OUT_CHANNELS=" not in args.upper() \
                            or "KERNEL_SIZE=" not in args.upper():
                        print(f"Required parameter missing for ConvTranspose2d() layer at line {idx}\n"
                              f"Provided: {line}\n"
                              f"Usage: ConvTranspose2d(in_channels=x, out_channels=y, kernel_size=z, "
                              f"(OPTIONAL: stride, padding, dilation)")
                        quit(1)

                    args = args.split(",")

                    # Initiate and setup arguments

                    in_channels, out_channels, kernel_size = None, None, None
                    stride, padding, dilation = None, None, None

                    for argument in args:
                        if argument.count("=") != 1:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Missing '=' or ','.")
                            quit(1)
                        try:
                            match argument[0:argument.index("=")].replace(" ", "").upper():
                                case "IN_CHANNELS":
                                    in_channels = int(argument[argument.index("=") + 1:])
                                case "OUT_CHANNELS":
                                    out_channels = int(argument[argument.index("=") + 1:])
                                case "KERNEL_SIZE":
                                    kernel_size = int(argument[argument.index("=") + 1:])
                                case "STRIDE":
                                    stride = int(argument[argument.index("=") + 1:])
                                case "PADDING":
                                    padding = int(argument[argument.index("=") + 1:])
                                case "DILATION":
                                    dilation = int(argument[argument.index("=") + 1:])
                                case _:
                                    print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                          f"Unknown argument '{argument}' provided.\n"
                                          f"Usage: ConvTranspose2d(in_channels=x, out_channels=y, kernel_size=z, "
                                          f"(OPTIONAL: stride, padding, dilation)")
                                    quit(1)
                        except ValueError:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Incorrect value provided for argument. All values must be integers.")
                            quit(1)

                    layer_stack.append(call_with_defaults(nn.ConvTranspose2d, in_channels=in_channels,
                                                          out_channels=out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=padding, dilation=dilation))

                # POOLING LAYERS =======================================================================================

                case "MAXPOOL2D":
                    if "KERNEL_SIZE=" not in args.upper():
                        print(f"Required parameter missing for MaxPool2d() layer at line {idx}\n"
                              f"Provided: {line}\n"
                              f"Usage: MaxPool2d(kernel_size=x, (OPTIONAL: stride, padding, dilation)")
                        quit(1)

                    args = args.split(",")

                    # Initiate and setup arguments
                    kernel_size, stride, padding, dilation = None, None, None, None

                    for argument in args:
                        if argument.count("=") != 1:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Missing '=' or ','.")
                            quit(1)
                        try:
                            match argument[0:argument.index("=")].replace(" ", "").upper():
                                case "KERNEL_SIZE":
                                    kernel_size = int(argument[argument.index("=") + 1:])
                                case "STRIDE":
                                    stride = int(argument[argument.index("=") + 1:])
                                case "PADDING":
                                    padding = int(argument[argument.index("=") + 1:])
                                case "DILATION":
                                    dilation = int(argument[argument.index("=") + 1:])
                                case _:
                                    print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                          f"Unknown argument '{argument}' provided.\n"
                                          f"Usage: MaxPool2d(kernel_size=x, (OPTIONAL: stride, padding, dilation)")
                                    quit(1)
                        except ValueError:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Incorrect value provided for argument. All values must be integers.")
                            quit(1)

                    layer_stack.append(call_with_defaults(nn.MaxPool2d, kernel_size=kernel_size, stride=stride,
                                                          padding=padding, dilation=dilation))

                case "AVGPOOL2D":
                    if "KERNEL_SIZE=" not in args.upper():
                        print(f"Required parameter missing for AvgPool2d() layer at line {idx}\n"
                              f"Provided: {line}\n"
                              f"Usage: AvgPool2d(kernel_size=x, (OPTIONAL: stride, padding)")
                        quit(1)

                    args = args.split(",")

                    # Initiate and setup arguments
                    kernel_size, stride, padding = None, None, None

                    for argument in args:
                        if argument.count("=") != 1:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Missing '=' or ','.")
                            quit(1)
                        try:
                            match argument[0:argument.index("=")].replace(" ", "").upper():
                                case "KERNEL_SIZE":
                                    kernel_size = int(argument[argument.index("=") + 1:])
                                case "STRIDE":
                                    stride = int(argument[argument.index("=") + 1:])
                                case "PADDING":
                                    padding = int(argument[argument.index("=") + 1:])
                                case _:
                                    print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                          f"Unknown argument '{argument}' provided.\n"
                                          f"Usage: AvgPool2d(kernel_size=x, (OPTIONAL: stride, padding)")
                                    quit(1)
                        except ValueError:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Incorrect value provided for argument. All values must be integers.")
                            quit(1)

                    layer_stack.append(call_with_defaults(nn.AvgPool2d, kernel_size=kernel_size, stride=stride,
                                                          padding=padding))

                case "ADAPTIVEAVGPOOL2D":
                    if "OUTPUT_SIZE=" not in args.upper():
                        print(f"Required parameter missing for MaxPool2d() layer at line {idx}\n"
                              f"Provided: {line}\n"
                              f"Usage: AdaptiveAvgPool2d(output_size=(x,y))")
                        quit(1)

                    args = args.lower()
                    args = args.replace("(", "")
                    args = args.replace(")", "")
                    args = args.replace("[", "")
                    args = args.replace("]", "")
                    args = args.replace("output_size=", "")
                    args = args.split(",")

                    if len(args) != 2:
                        print(f"Incorrect parameters provided for MaxPool2d() layer at line {idx}\n"
                              f"Provided: {line}\n"
                              f"Usage: AdaptiveAvgPool2d(output_size=(x,y))")
                        quit(1)

                    try:
                        layer_stack.append(nn.AdaptiveAvgPool2d((int(args[0]), int(args[1]))))
                    except ValueError:
                        print(f"Error while decoding arguments in '{line}' at line {idx}.\n"
                              f"Incorrect value provided for argument. All values must be integers.\n"
                              f"Usage: AdaptiveAvgPool2d(output_size=(x,y))")
                        quit(1)

                # NORMALIZATION LAYERS =================================================================================
                case "BATCHNORM2D":
                    if "NUM_FEATURES=" not in args.upper():
                        print(f"Required parameter missing for BatchNorm2d() layer at line {idx}\n"
                              f"Provided: {line}\n"
                              f"Usage: BatchNorm2d(num_features=x, (OPTIONAL: eps, momentum, affine)")
                        quit(1)

                    args = args.split(",")

                    # Initiate and setup arguments
                    num_features, eps, momentum, affine = None, None, None, None

                    for argument in args:
                        if argument.count("=") != 1:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Missing '=' or ','.")
                            quit(1)
                        try:
                            match argument[0:argument.index("=")].replace(" ", "").upper():
                                case "NUM_FEATURES":
                                    num_features = int(argument[argument.index("=") + 1:])
                                case "EPS":
                                    eps = float(argument[argument.index("=") + 1:])
                                case "MOMENTUM":
                                    momentum = float(argument[argument.index("=") + 1:])
                                case "AFFINE":
                                    if argument[argument.index("=") + 1:].upper() not in ["TRUE", "FALSE"]:
                                        raise ValueError
                                    affine = False if argument[argument.index("=") + 1:].upper() == "FALSE" else True
                                case _:
                                    print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                          f"Unknown argument '{argument}' provided.\n"
                                          f"Usage: BatchNorm2d(num_features=x, (OPTIONAL: eps, momentum, affine)")
                                    quit(1)
                        except ValueError:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Incorrect value provided for argument.\n"
                                  f"Usage: BatchNorm2d(num_features=x (int), (OPTIONAL: eps (float), "
                                  f"momentum (float), affine (bool))")
                            quit(1)

                    layer_stack.append(call_with_defaults(nn.BatchNorm2d, num_features=num_features, eps=eps,
                                                          momentum=momentum, affine=affine))

                case "GROUPNORM":
                    if "NUM_GROUPS=" not in args.upper() or "NUM_CHANNELS=" not in args.upper():
                        print(f"Required parameter missing for GroupNorm() layer at line {idx}\n"
                              f"Provided: {line}\n"
                              f"Usage: GroupNorm(num_groups=x, num_channels=y,(OPTIONAL: eps, affine)")
                        quit(1)

                    args = args.split(",")

                    # Initiate and setup arguments
                    num_groups, num_channels, eps, affine = None, None, None, None

                    for argument in args:
                        if argument.count("=") != 1:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Missing '=' or ','.")
                            quit(1)
                        try:
                            match argument[0:argument.index("=")].replace(" ", "").upper():
                                case "NUM_GROUPS":
                                    num_groups = int(argument[argument.index("=") + 1:])
                                case "EPS":
                                    eps = float(argument[argument.index("=") + 1:])
                                case "NUM_CHANNELS":
                                    num_channels = int(argument[argument.index("=") + 1:])
                                case "AFFINE":
                                    affine = False if argument[argument.index("=") + 1:].upper() == "FALSE" else True
                                case _:
                                    print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                          f"Unknown argument '{argument}' provided.\n"
                                          f"Usage: GroupNorm(num_groups=x, nun_channels=y, (OPTIONAL: eps, affine)")
                                    quit(1)
                        except ValueError:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Incorrect value provided for argument.\n"
                                  f"Usage: GroupNorm2d(num_groups=x (int), num_channels=y (int),  "
                                  f"(OPTIONAL: eps (float), affine (bool))")
                            quit(1)

                    layer_stack.append(call_with_defaults(nn.GroupNorm, num_groups=num_groups, num_channels = num_channels,
                                                          eps=eps, affine=affine))

                # ACTIVATION FUNCTIONS =================================================================================
                case "RELU":
                    layer_stack.append(nn.ReLU())

                case "LEAKYRELU":
                    layer_stack.append(nn.LeakyReLU())

                case "GELU":
                    layer_stack.append(nn.GELU())

                case "SILU":
                    layer_stack.append(nn.SiLU())

                case "SIGMOID":
                    layer_stack.append(nn.Sigmoid())

                # DROPOUT AND REGULARIZATION ===========================================================================
                case "DROPOUT":
                    if "p=" in args:
                        try:
                            p = float(args[args.index("=") + 1:])
                        except ValueError:
                            print(f"ERROR while processing line {idx} in model template.\n"
                                  f"Bad argument provided {args}.\n"
                                  f"Usage Dropout(p=0.5)")

                        layer_stack.append(nn.Dropout(p))
                    else:
                        layer_stack.append(nn.Dropout())

                case "DROPOUT2D":
                    if "p=" in args:
                        try:
                            p = float(args[args.index("=") + 1:])
                        except ValueError:
                            print(f"ERROR while processing line {idx} in model template.\n"
                                  f"Bad argument provided {args}.\n"
                                  f"Usage Dropout(p=0.5)")

                        layer_stack.append(nn.Dropout2d(p))
                    else:
                        layer_stack.append(nn.Dropout2d())

                case "ALPHADROPOUT":
                    layer_stack.append(nn.AlphaDropout())




                # FULLY CONNECTED AND OUTPUT ===========================================================================

                case "FLATTEN":
                    layer_stack.append(nn.Flatten())

                case "LINEAR":

                    if "IN_FEATURES=" not in args.upper() or "OUT_FEATURES=" not in args.upper():
                        print(f"Required parameter missing for Linear() layer at line {idx}\n"
                              f"Provided: {line}\n"
                              f"Usage: Linear(in_features=x, out_features=y)")
                        quit(1)

                    args = args.split(",")

                    # Initiate and setup arguments
                    in_features, out_features = None, None
                    for argument in args:
                        if argument.count("=") != 1:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Missing '=' or ','.")
                            quit(1)
                        try:
                            match argument[0:argument.index("=")].replace(" ", "").upper():
                                case "IN_FEATURES":
                                    in_features = int(argument[argument.index("=") + 1:])
                                case "OUT_FEATURES":
                                    out_features = int(argument[argument.index("=") + 1:])
                                case _:
                                    print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                          f"Unknown argument '{argument}' provided.\n"
                                          f"Usage: Linear(in_features=x, out_features=y)")
                                    quit(1)
                        except ValueError:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Incorrect value provided for argument. All values must be integers.")
                            quit(1)

                    layer_stack.append(nn.Linear(in_features=in_features, out_features=out_features))

                case "LAZYLINEAR":

                    if "OUT_FEATURES=" not in args.upper():
                        print(f"Required parameter missing for LazyLinear() layer at line {idx}\n"
                              f"Provided: {line}\n"
                              f"Usage: LazyLinear(out_features=x)")
                        quit(1)

                    args = args.split(",")

                    # Initiate and setup arguments
                    out_features = None

                    for argument in args:
                        if argument.count("=") != 1:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Missing '=' or ','.")
                            quit(1)
                        try:
                            match argument[0:argument.index("=")].replace(" ", "").upper():
                                case "OUT_FEATURES":
                                    out_features = int(argument[argument.index("=") + 1:])
                                case _:
                                    print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                          f"Unknown argument '{argument}' provided.\n"
                                          f"Usage: LazyLinear(out_features=y)")
                                    quit(1)
                        except ValueError:
                            print(f"Error while decoding argument '{argument}' at line {idx}.\n"
                                  f"Incorrect value provided for argument. All values must be integers.")
                            quit(1)

                    layer_stack.append(nn.LazyLinear(out_features=out_features))


                case _:
                    print(f"ERROR while decoding model template at {path}\n"
                          f"Unknown parameter '{line}' provided at line {idx}.")

                    print("Available Layers:\n"
                          "Conv2d, ConvTranspose2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, BatchNorm2d,\n"
                          "GroupNorm", "ReLU", "LeakyReLU", "GeLU", "SiLU", "Dropout", "Dropout2d", "AlphaDropout\n",
                          "Flatten", "Linear")

                    print("Look up torch.nn documentation for details.")
                    quit(1)


    return layer_stack


def generate_sequential_stack_from_template(template_path):
    template_path = Path(template_path).resolve()

    if not template_path.exists():
        print("ERROR while creating model from template.\n"
              f"Given path {template_path} does not exist")
        quit()
    
    
    
    layer_stack = parse_model_template(template_path)

    sequential_stack = nn.Sequential(
        *layer_stack
    )

    print(f"Successfully created layer stack from template file at {template_path}")
    return sequential_stack