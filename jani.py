from __future__ import annotations
import json
import copy
import random

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Union


@dataclass
class Action:
    label: str
    idx: int

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class Variable:
    name: str
    idx: int
    type: str
    value: Union[int, float, bool]
    kind: Union[str, None] = None
    upper_bound: Union[int, float, None] = None
    lower_bound: Union[int, float, None] = None


    def __hash__(self) -> int:
        return hash(self.name)

    def random(self) -> None:
        if self.type == 'int':
            assert self.lower_bound is not None and self.upper_bound is not None
            assert isinstance(self.lower_bound, int) and isinstance(self.upper_bound, int)
            self.value = random.randint(self.lower_bound, self.upper_bound)
        elif self.type == 'real':
            assert self.lower_bound is not None and self.upper_bound is not None
            assert isinstance(self.lower_bound, float) and isinstance(self.upper_bound, float)
            self.value = random.uniform(self.lower_bound, self.upper_bound)
        elif self.type == 'bool':
            self.value = random.choice([True, False])
        else:
            raise ValueError(f'Unsupported variable type: {self.type}')


# Constant is a special type of variable
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
    def construct(expr: Union[dict, str, int, float, bool]) -> Expression:
        """Construct an expression from a JSON object recursively."""
        if isinstance(expr, str):
            return VarExpression(expr)
        elif isinstance(expr, (int, float, bool)):
            return ConstantExpression(expr)
        elif expr['op'] == '+':
            return AddExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        elif expr['op'] == '-':
            return SubExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        elif expr['op'] == '*':
            return MulExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        elif expr['op'] == '∧':
            return ConjExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        elif expr['op'] == '∨':
            return DisjExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        elif expr['op'] == '=':
            return EqExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        elif expr['op'] == '≤':
            return LeExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
        else:
            raise ValueError(f'Unsupported operator: {expr["op"]}')


class VarExpression(Expression):
    def __init__(self, variable: str):
        self.variable = variable

    def evaluate(self, state: State) -> float:
        return state[self.variable].value


class ConstantExpression(Expression):
    def __init__(self, value: Union[int, float, bool]):
        self.value = value

    def evaluate(self, state: State) -> Union[int, float, bool]:
        return self.value
    

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
    

class DivExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, state: State) -> float:
        return self.left.evaluate(state) / self.right.evaluate(state)
    

class ConjExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) and self.right.evaluate(state)


class DisjExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) or self.right.evaluate(state)


class EqExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) == self.right.evaluate(state)
    

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

    def apply(self, state: State) -> tuple[list[State], list[float]]:
        if self._guard.evaluate(state):
            new_states = []
            distribution = []
            for destination in self._destinations:
                new_state = copy.deepcopy(state)
                for assignment in destination.assignments:
                    new_state[assignment.target] = assignment.value.evaluate(state)
                new_states.append(new_state)
                distribution.append(destination.probability)
            assert sum(distribution) == 1.0
            return (new_states, distribution)
        else:
            return ([], [])


class Automaton:
    def __init__(self, json_obj: dict):
        def create_edge_dict() -> dict[str, list[Edge]]:
            """Create a dictionary of edges, indexed by the action label."""
            edge_dict = defaultdict(list)
            for edge in json_obj['edges']:
                edge_obj = Edge(edge)
                edge_dict[edge_obj._label].append(edge_obj)
            return edge_dict

        self._name: str = json_obj['name']
        self._edges = create_edge_dict()
        # So far, we won't use the following fields
        self._initial_locations: list[str] = json_obj['initial-locations']
        self._locations: list[dict[str, str]] = json_obj['locations']

    def transit(self, state: State, action: Action) -> list[State]:
        """Apply the given action to the given state."""
        if action.label not in self._edges:
            raise ValueError(f'Action {action.label} is not supported in automaton {self._name}.')
        new_states = []
        for edge in self._edges[action.label]:
            if not edge.is_enabled(state):
                continue
            successors, distribution = edge.apply(state)
            next_state = random.choices(successors, distribution)[0]
            new_states.append(next_state)
        return new_states


class JANI:
    def __init__(self, file_path: str):
        def add_action(action_info: dict, idx: int) -> Action:
            """Add a new action to the action list."""
            return Action(action_info['name'], idx)

        def add_variable(variable_info: dict, idx: int) -> Variable:
            """Add a new variable to the variable list."""
            properties: dict = variable_info['type']
            if not properties['kind'] == 'bounded':
                raise ValueError(f'Unsupported variable kind: {properties["kind"]}')
            variable_type = properties['base']
            variable_kind = properties['kind']
            if variable_type == 'int':
                lower_bound, upper_bound = int(properties['lower-bound']), int(properties['upper-bound'])
            elif variable_type == 'real':
                lower_bound, upper_bound = float(properties['lower-bound']), float(properties['upper-bound'])
            elif variable_type == 'bool':
                lower_bound, upper_bound = None, None
            else:
                raise ValueError(f'Unsupported variable type: {variable_type}')
            return Variable(variable_info['name'], idx, variable_type, variable_info['initial-value'], variable_kind, lower_bound, upper_bound)
        
        def add_constant(constant_info: dict, idx: int) -> Constant:
            """Add a new constant to the constant list."""
            name = constant_info['name']
            value = constant_info['value']
            constant_type = constant_info['type']
            if constant_info['type'] == 'int':
                value = int(value)
            elif constant_info['type'] == 'real':
                value = float(value)
            elif constant_info['type'] == 'bool':
                value = value == 'true'
            else:
                raise ValueError(f'Unsupported constant type: {constant_info["type"]}')
            return Variable(name, idx, constant_type, value)
        
        jani_obj = json.loads(Path(file_path).read_text('utf-8'))
        # extract actions, constants, and variables
        self._actions: list[Action] = [add_action(action, idx) for idx, action in enumerate(jani_obj['actions'])]
        self._constants: list[Constant] = [add_constant(constant, idx) for idx, constant in enumerate(jani_obj['constants'])]
        self._variables: list[Variable] = [add_variable(variable, idx + len(self._constants)) for idx, variable in enumerate(jani_obj['variables'])]
        self._automata: list[Automaton] = [Automaton(automaton) for automaton in jani_obj['automata']]
        if len(self._automata) > 1:
            raise ValueError('Multiple automata are not supported yet.')