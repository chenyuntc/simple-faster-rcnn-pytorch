def get_dataset_directory(dataset_name, create_directory=True):
    """Gets the path to the directory of given dataset.

    The generated path is just a concatenation of the global root directory
    (see :func:`set_dataset_root` for how to change it) and the dataset name.
    The dataset name can contain slashes, which are treated as path separators.

    Args:
        dataset_name (str): Name of the dataset.
        create_directory (bool): If True (default), this function also creates
            the directory at the first time. If the directory already exists,
            then this option is ignored.

    Returns:
        str: Path to the dataset directory.

    """
    path = os.path.join(_dataset_root, dataset_name)
    if create_directory:
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return path


import numpy as np
from PIL import Image


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))
