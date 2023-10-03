import json
from jsonpath_ng import jsonpath, parse

class BaseModel:
    def inovke_api(self):
        raise NotImplementedError("Not implemented in base model, make sure to override this method.")