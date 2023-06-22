import os


def get_images(directory, extension: str = ".png") -> list:
    png_files = []

    return [
        os.path.join(dirpath, filename)
        for dirpath, dirnames, filenames in os.walk(directory)
        for filename in filenames
        if filename.endswith(extension)
    ]
