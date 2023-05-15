import os
import json
import sys

from pydantic.main import BaseModel

import numaprom


def generate_module_schemas(module, output_dir: str):
    schemas = {}

    for name in dir(module):
        cls = getattr(module, name)
        if isinstance(cls, type) and issubclass(cls, BaseModel):
            schema = cls.schema()
            schemas[name] = schema

    for name, schema in schemas.items():
        with open(os.path.join(output_dir, f'{name}.json'), 'w') as f:
            json.dump(schema, f, indent=2)

    return schemas


if __name__ == "__main__":
    schema_dir = sys.argv[1]
    generate_module_schemas(numaprom, schema_dir)

