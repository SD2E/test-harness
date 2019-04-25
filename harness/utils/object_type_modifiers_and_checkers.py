from six import string_types


def make_list_if_not_list(obj):
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


def is_list_of_strings(obj):
    if obj and isinstance(obj, list):
        return all(isinstance(elem, string_types) for elem in obj)
    else:
        return False
