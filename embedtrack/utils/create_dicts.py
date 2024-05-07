"""
Original work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler, Karlsruhe Institute of Technology (MIT License)
Modifications: remove 3d; remove one hot; change augmentation; change create_model_dict
"""


def create_model_dict(input_channels, n_classes, name="2d"):
    """
    Creates `model_dict` dictionary from parameters.
    Parameters
    ----------
    input_channels: int
        1 indicates gray-channle image, 3 indicates RGB image.
    num_classes: list
        [4, 1] -> 4 indicates offset in x, offset in y, margin in x, margin in y; 1 indicates seediness score
    name: string
    """
    model_dict = {
        "name": "TrackERFNet"
        if name == "2d"
        else AssertionError(f"Unknown dimensions {name}"),
        "kwargs": {
            "n_classes": n_classes,
            "input_channels": input_channels,
        },
    }
    print(
        "`model_dict` dictionary successfully created with: \n -- num of classes equal to {}, \n -- input channels equal to {}, \n -- name equal to {}".format(
            n_classes, input_channels, name
        )
    )
    return model_dict