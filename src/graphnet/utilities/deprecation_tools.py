"""Utility functions for handling deprecation transitions."""

from typing import Dict, Tuple
from copy import deepcopy
from torch import Tensor


def rename_state_dict_entries(
    state_dict: Dict[str, Tensor], old_phrase: str, new_phrase: str
) -> Tuple[Dict[str, Tensor], bool]:
    """Replace `old_phrase` in state dict fields with `new_phrase`.

    Returned state dict is a deepcopy of the input.

    Args:
        state_dict: The state dict whos fields need renaming.
        old_phrase: Phrase in state dict field that needs to be replaced.
        new_phrase: Phrase to add in place of `old_phrase` in state dict.
    """
    assert isinstance(old_phrase, str)
    assert isinstance(new_phrase, str)

    # Make a carbon-copy
    new_state_dict = deepcopy(state_dict)

    # Replace old entries in copy
    state_dict_altered = False
    for key in state_dict.keys():
        if old_phrase in key:
            new_key = key.replace(old_phrase, new_phrase)
            new_state_dict[new_key] = new_state_dict.pop(key)
            state_dict_altered = True

    return new_state_dict, state_dict_altered
