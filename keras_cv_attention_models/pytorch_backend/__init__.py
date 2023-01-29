import inspect

_GLOBAL_CUSTOM_OBJECTS = {}
_GLOBAL_CUSTOM_NAMES = {}

def image_data_format():
    return "channels_first"


def register_keras_serializable(package="Custom", name=None):
    def decorator(arg):
        """Registers a class with the Keras serialization framework."""
        class_name = name if name is not None else arg.__name__
        registered_name = package + ">" + class_name

        if inspect.isclass(arg) and not hasattr(arg, "get_config"):
            raise ValueError(
                "Cannot register a class that does not have a "
                "get_config() method."
            )

        if registered_name in _GLOBAL_CUSTOM_OBJECTS:
            raise ValueError(
                f"{registered_name} has already been registered to "
                f"{_GLOBAL_CUSTOM_OBJECTS[registered_name]}"
            )

        if arg in _GLOBAL_CUSTOM_NAMES:
            raise ValueError(
                f"{arg} has already been registered to "
                f"{_GLOBAL_CUSTOM_NAMES[arg]}"
            )
        _GLOBAL_CUSTOM_OBJECTS[registered_name] = arg
        _GLOBAL_CUSTOM_NAMES[arg] = registered_name

        return arg

    return decorator
