import json

from pathlib import Path
from dataclasses import dataclass


@dataclass
class Action:
    name: str
    idx: int

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class Variable:
    name: str
    idx: int
    value: float

    def __hash__(self) -> int:
        return hash(self.name)


class JANI:
    def __init__(self, file_path: str):
        def add_action(action_info: dict, idx: int) -> Action:
            """Add a new action to the action list."""
            return Action(action_info['name'], idx)

        def add_variable(variable_info: dict, idx: int) -> Variable:
            """Add a new variable to the variable list."""
            return Variable(variable_info['name'], idx, variable_info['initial-value'])

        jani_obj = json.loads(Path(file_path).read_text('utf-8'))
        # extract actions and variables
        self._actions = [add_action(action, idx) for idx, action in enumerate(jani_obj['actions'])]
        self._variables = [add_variable(variable, idx) for idx, variable in enumerate(jani_obj['variables'])]