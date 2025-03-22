from enum import Enum
# Option definitions
class BackendOption(Enum):
    C_EXTENSION = 1
    PYTHON = 2

# Config declarations
backend_option = BackendOption.PYTHON