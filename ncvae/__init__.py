import os
import importlib

# Get the current package's path
package_dir = os.path.dirname(__file__)

# Iterate over all .py files except __init__.py
for filename in os.listdir(package_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # strip .py
        importlib.import_module(f"{__name__}.{module_name}")

