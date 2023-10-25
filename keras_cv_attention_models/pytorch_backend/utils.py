import os
import inspect
import torch
import hashlib

_GLOBAL_CUSTOM_OBJECTS = {}
_GLOBAL_CUSTOM_NAMES = {}


def register_keras_serializable(package="Custom", name=None):
    def decorator(arg):
        """Registers a class with the Keras serialization framework."""
        class_name = name if name is not None else arg.__name__
        registered_name = package + ">" + class_name

        if inspect.isclass(arg) and not hasattr(arg, "get_config"):
            raise ValueError("Cannot register a class that does not have a " "get_config() method.")

        # if registered_name in _GLOBAL_CUSTOM_OBJECTS:
        #     raise ValueError(f"{registered_name} has already been registered to " f"{_GLOBAL_CUSTOM_OBJECTS[registered_name]}")

        # if arg in _GLOBAL_CUSTOM_NAMES:
        #     raise ValueError(f"{arg} has already been registered to " f"{_GLOBAL_CUSTOM_NAMES[arg]}")
        _GLOBAL_CUSTOM_OBJECTS[registered_name] = arg
        _GLOBAL_CUSTOM_NAMES[arg] = registered_name

        return arg

    return decorator


def validate_file_md5(fpath, file_hash, chunk_size=65535):
    """Validates a file against a md5 hash. From keras/utils/data_utils.py"""
    hasher = hashlib.md5()
    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)
    return str(hasher.hexdigest()) == str(file_hash)


def _extract_archive(file_path, path=".", archive_format="auto"):
    if "zip" in archive_format or (archive_format == "auto" and file_path.endswith(".zip")):
        import zipfile

        assert zipfile.is_zipfile(file_path), "Not a zip file: {}".format(file_path)
        open_fn = zipfile.ZipFile
    elif "tar" in archive_format or (archive_format == "auto" and (file_path.endswith(".tar") or file_path.endswith(".tar.gz"))):
        import tarfile

        assert tarfile.is_tarfile(file_path), "Not a tar file: {}".format(file_path)
        open_fn = tarfile.open
    else:
        raise ValueError("Not a supported extract file format: {}".format(file_path))

    print(">>>> Extract {} -> {}".format(file_path, path))
    with open_fn(file_path) as ff:
        ff.extractall(path)
    return path


def get_file(fname=None, origin=None, cache_subdir="datasets", file_hash=None, extract=False):
    # print(f">>>> {fname = }, {origin = }, {cache_subdir = }, {file_hash = }")
    save_dir = os.path.join(os.path.expanduser("~"), ".keras", cache_subdir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    fname = os.path.basename(origin) if fname is None else fname
    file_path = os.path.join(save_dir, fname)
    if os.path.exists(file_path):
        if file_hash is not None and not validate_file_md5(file_path, file_hash):
            print(
                "A local file was found, but it seems to be incomplete or outdated because the md5 file hash does not match the original value of "
                f"{file_hash} so we will re-download the data."
            )
        else:
            return file_path

    print("Downloading data from {} to {}".format(origin, file_path))
    torch.hub.download_url_to_file(origin, file_path)
    if os.path.exists(file_path) and file_hash is not None and not validate_file_md5(file_path, file_hash):
        raise ValueError("Incomplete or corrupted file detected. The md5 file hash does not match the provided value {}.".format(file_hash))

    if extract:
        _extract_archive(file_path, path=save_dir)  # return tar file path, just like keras one
    return file_path
