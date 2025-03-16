from enum import Enum
# Option definitions
class BackendOption(Enum):
    C_EXTENSION = 1
    PYTHON = 2
    NUMPY = 3

# Config declarations
backend_option = BackendOption.C_EXTENSION