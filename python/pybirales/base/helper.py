def to_string(value):
    if type(value) == bytes:
        return value.decode("utf-8")
    elif type(value) == str:
        return value
    else:
        return str(value)
