import os
import inspect
import torch

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


def get_file(fname=None, origin=None, cache_subdir="datasets", file_hash=None):
    # print(f">>>> {fname = }, {origin = }, {cache_subdir = }, {file_hash = }")
    save_dir = os.path.join(os.path.expanduser("~/.keras"), cache_subdir)
    if os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    fname = os.path.basename(origin) if fname is None else fname
    file_path = os.path.join(save_dir, fname)
    if os.path.exists(file_path):
        return file_path  # [TODO] check md5
    torch.hub.download_url_to_file(origin, file_path)
    return file_path
