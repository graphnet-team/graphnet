from enum import Enum

class _SUPPORTED_MODES:
    inference = 'inference'
    oscNext = 'oscNext'
    NuGen = 'NuGen'

print([e for e in _SUPPORTED_MODES])
print(_SUPPORTED_MODES.inference == 'inference')