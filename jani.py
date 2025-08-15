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
        elif expr['op'] == '<':
            return LtExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
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


class LtExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) < self.right.evaluate(state)
    

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
                # Handle both 'target' and 'ref' field names for assignment target
                target_field = assignment.get('target') or assignment.get('ref')
                assignments.append(Assignment(target_field, Expression.construct(assignment['value'])))
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
    def __init__(self, file_path: str, start_file: str, goal_file: str, failure_file: str):
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
            return Variable(variable_info['name'], idx, variable_type, variable_info['initial-value'], variable_kind, upper_bound, lower_bound)
        
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

        def init_state_generator(json_obj: dict) -> JANI.InitGenerator:
            if json_obj['op'] == "states-values":
                return JANI.FixedGenerator(json_obj, self)
            raise ValueError(f"Unsupported init state generator operation: {json_obj['op']}")

        def goal_expression(json_obj: dict) -> Expression:
            if json_obj['op'] == "objective":
                goal_expr = json_obj['goal']['exp']
                if goal_expr['op'] == "state-condition":
                    return Expression.construct(goal_expr['exp'])
                else:
                    raise ValueError(f"Unsupported goal expression operation: {goal_expr['op']}")
            else:
                raise ValueError(f"Unsupported goal expression operation: {json_obj['op']}")

        def failure_expression(json_obj: dict) -> Expression:
            if json_obj['op'] == "state-condition":
                return Expression.construct(json_obj['exp'])
            else:
                raise ValueError(f"Unsupported safe expression operation: {json_obj['op']}")

        jani_obj = json.loads(Path(file_path).read_text('utf-8'))
        # extract actions, constants, and variables
        self._actions: list[Action] = [add_action(action, idx) for idx, action in enumerate(jani_obj['actions'])]
        self._constants: list[Constant] = [add_constant(constant, idx) for idx, constant in enumerate(jani_obj['constants'])]
        self._variables: list[Variable] = [add_variable(variable, idx + len(self._constants)) for idx, variable in enumerate(jani_obj['variables'])]
        self._automata: list[Automaton] = [Automaton(automaton) for automaton in jani_obj['automata']]
        if len(self._automata) > 1:
            raise ValueError('Multiple automata are not supported yet.')
        # start states
        start_spec = json.loads(Path(start_file).read_text('utf-8'))
        self._init_generator = init_state_generator(start_spec)
        # goal states
        goal_spec = json.loads(Path(goal_file).read_text('utf-8'))
        self._goal_expr = goal_expression(goal_spec)
        # failure condition
        failure_spec = json.loads(Path(failure_file).read_text('utf-8'))
        self._failure_expr = failure_expression(failure_spec)

    class InitGenerator(ABC):
        '''Generate initial states.'''
        @abstractmethod
        def generate(self) -> State:
            pass

    class FixedGenerator(InitGenerator):
        '''Generate a fixed set of initial states.'''
        def __init__(self, json_obj: dict, model: JANI):
            def create_state(state_value: list[dict]) -> State:
                variable_dict = {variable_info['var']: variable_info['value'] for variable_info in state_value['variables']}
                # Copy the constants
                state_dict = {}
                for constant in model._constants:
                    state_dict[constant.name] = copy.deepcopy(constant)
                for _variable in model._variables:
                    variable = copy.deepcopy(_variable)
                    value = variable_dict[variable.name]
                    variable.value = value
                    state_dict[variable.name] = variable
                return State(state_dict)

            self._pool: list[State] = []
            for state_value in json_obj['values']:
                state = create_state(state_value)
                self._pool.append(state)
      
        def generate(self) -> State:
            return random.choice(self._pool)
        
    def reset(self) -> State:
        """Reset the JANI model to a random initial state."""
        return self._init_generator.generate()
    
    def get_action_count(self) -> int:
        return len(self._actions)
    
    def get_constants_variables(self) -> list[Variable]:
        return self._constants + self._variables
    
    def get_transition(self, state: State, action: Action) -> State:
        # Implement the logic to get the next state based on the current state and action
        next_states = self._automata[0].transit(state, action)
        if len(next_states) == 0:
            return None
        if len(next_states) > 1:
            print(f"Warning: Multiple next states found for action {action.label}. Choosing the first one.")
        return next_states[0]

    def goal_reached(self, state: State) -> bool:
        # Implement the logic to check if the goal state is reached
        return self._goal_expr.evaluate(state)

    def failure_reached(self, state: State) -> bool:
        # Implement the logic to check if the failure state is reached
        return self._failure_expr.evaluate(state)