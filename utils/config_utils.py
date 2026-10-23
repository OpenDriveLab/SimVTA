import runpy
from types import SimpleNamespace

class Config(SimpleNamespace):
    @classmethod
    def fromfile(cls, path):
        return cls(**{k: v for k, v in runpy.run_path(str(path)).items() if not k.startswith("__")})

    def merge_from_dict(self, values):
        self.__dict__.update(values)
