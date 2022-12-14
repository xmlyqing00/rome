

class DictObj:

    def __init__(self, in_dict: dict):
        for key, val in in_dict.items():
            if isinstance(val, dict):
                setattr(self, key, DictObj(val))
            else:
                setattr(self, key, val)

    def __str__(self) -> str:
        return str(vars(self))