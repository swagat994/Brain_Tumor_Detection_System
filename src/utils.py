import os


def create_directory(path):
    """
    Create a directory if it does not exist.
    """

    os.makedirs(path, exist_ok=True)


def get_file_size(file_path):
    """
    Return file size in bytes.
    """

    return os.path.getsize(file_path)