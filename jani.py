from __future__ import annotations
import json
import copy

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Union


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
    value: Union[int, float, bool]

    def __hash__(self) -> int:
        return hash(self.name)
    

# treat constants as variables
type Constant = Variable


@dataclass
class State:
    variable_dict: dict[str, Variable]

    def __getitem__(self, key: str) -> Variable:
        return self.variable_dict[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.variable_dict[key].value = value
    
    def to_vector(self) -> list[float]:
        '''Convert the state to a vector, according to the idx.'''
        vec = [0.0] * len(self.variable_dict)
        for variable in self.variable_dict.values():
            vec[variable.idx] = variable.value
        return vec


class Expression(ABC):
    @abstractmethod
    def evaluate(self, state: State) -> Any:
        pass

    @staticmethod
    def construct(expr: Union[dict, str]) -> Expression:
        """Construct an expression from a JSON object recursively."""
        if isinstance(expr, str):
            return VarExpression(expr)
        elif expr['op'] == '+':
            return AddExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        elif expr['op'] == '-':
            return SubExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        elif expr['op'] == '*':
            return MulExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        elif expr['op'] == '∧':
            return ConjExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        elif expr['op'] == '≤':
            return LeExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        else:
            raise ValueError(f'Unsupported operator: {expr["op"]}')


class VarExpression(Expression):
    def __init__(self, variable: str):
        self.variable = variable

    def evaluate(self, state: State) -> float:
        return state[self.variable].value
    

class AddExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, state: State) -> float:
        return self.left.evaluate(state) + self.right.evaluate(state)
    

class SubExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, state: State) -> float:
        return self.left.evaluate(state) - self.right.evaluate(state)
    

class MulExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, state: State) -> float:
        return self.left.evaluate(state) * self.right.evaluate(state)
    

class ConjExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) and self.right.evaluate(state)
    

class LeExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) <= self.right.evaluate(state)
    

@dataclass
class Assignment:
    target: str
    value: Expression


@dataclass
class Destination:
    assignments: list[Assignment]
    probability: float


class Edge:
    def __init__(self, json_obj: dict):
        self._label = json_obj['action']
        self._guard = Expression.construct(json_obj['guard']['exp'])
        self._destinations = []
        for destination in json_obj['destinations']:
            assignments = []
            for assignment in destination['assignments']:
                assignments.append(Assignment(assignment['target'], Expression.construct(assignment['value'])))
            if 'probability' in destination:
                probability = destination['probability']
            else:
                probability = 1.0
            self._destinations.append(Destination(assignments, probability))

    def is_enabled(self, state: State) -> bool:
        return self._guard.evaluate(state)

    def apply(self, state: State) -> list[State]:
        if self._guard.evaluate(state):
            new_states = []
            for destination in self._destinations:
                new_state = copy.deepcopy(state)
                for assignment in destination.assignments:
                    new_state[assignment.target] = assignment.value.evaluate(state)
                new_states.append(new_state)
            return new_states
        else:
            return []


class JANI:
    def __init__(self, file_path: str):
        def add_action(action_info: dict, idx: int) -> Action:
            """Add a new action to the action list."""
            return Action(action_info['name'], idx)

        def add_variable(variable_info: dict, idx: int) -> Variable:
            """Add a new variable to the variable list."""
            return Variable(variable_info['name'], idx, variable_info['initial-value'])
        
        def add_constant(constant_info: dict, idx: int) -> Constant:
            """Add a new constant to the constant list."""
            name = constant_info['name']
            value = constant_info['value']
            if constant_info['type'] == 'int':
                value = int(value)
            elif constant_info['type'] == 'real':
                value = float(value)
            elif constant_info['type'] == 'bool':
                value = value == 'true'
            else:
                raise ValueError(f'Unsupported constant type: {constant_info["type"]}')
            return Constant(name, idx, value)

        jani_obj = json.loads(Path(file_path).read_text('utf-8'))
        # extract actions, constants, and variables
        self._actions = [add_action(action, idx) for idx, action in enumerate(jani_obj['actions'])]
        self._constants = [add_constant(constant, idx) for idx, constant in enumerate(jani_obj['constants'])]
        self._variables = [add_variable(variable, idx + len(self._constants)) for idx, variable in enumerate(jani_obj['variables'])]