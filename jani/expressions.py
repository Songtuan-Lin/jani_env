from abc import ABC, abstractmethod
from typing import Any, Union, Optional

from .core import JANI, State


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
        elif expr['op'] == '/':
            return DivExpression(Expression.construct(expr['left']), Expression.construct(expr['right']))
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

    def __repr__(self):
        return self.variable

    def evaluate(self, state: State) -> float:
        return state[self.variable].value
    
    def to_clause(self, ctx: JANI) -> tuple[ExprRef, list[ExprRef], list[ExprRef]]:
        '''Convert the variable expression to a Z3 clause.'''
        variable = ctx.get_variable(self.variable)
        additional_clauses = []
        if variable.type == 'int' or variable.type == 'real':
            v = Real(variable.name) if variable.type == 'real' else Int(variable.name)
            if variable.lower_bound is None and variable.upper_bound is not None:
                expr = v <= variable.upper_bound
            elif variable.lower_bound is not None and variable.upper_bound is None:
                expr = v >= variable.lower_bound
            elif variable.lower_bound is not None and variable.upper_bound is not None:
                expr = And(v >= variable.lower_bound, v <= variable.upper_bound)
            else:
                if variable.constant:
                    expr = v == variable.value
                else:
                    raise ValueError(f'Variable {variable.name} has no bounds defined.')
            additional_clauses.append(expr)
        elif variable.type == 'bool':
            v = Bool(variable.name)
            if variable.constant:
                expr = v == bool(variable.value)
            additional_clauses.append(expr)
        else:
            raise ValueError(f'Unsupported variable type: {variable.type}')
        return v, additional_clauses, [v]


class ConstantExpression(Expression):
    def __init__(self, value: Union[int, float, bool]):
        self.value = value

    def __repr__(self):
        return str(self.value)

    def evaluate(self, state: State) -> Union[int, float, bool]:
        return self.value

    def to_clause(self, ctx: JANI) -> tuple[ExprRef, list[ExprRef], list[ExprRef]]:
        return self.value, [], []


class AddExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left.__repr__()} + {self.right.__repr__()})"

    def evaluate(self, state: State) -> float:
        return self.left.evaluate(state) + self.right.evaluate(state)

    def to_clause(self, ctx: JANI) -> tuple[ExprRef, list[ExprRef], list[ExprRef]]:
        left, left_addt, left_vars = self.left.to_clause(ctx)
        right, right_addt, right_vars = self.right.to_clause(ctx)
        return left + right, left_addt + right_addt, left_vars + right_vars


class SubExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left.__repr__()} - {self.right.__repr__()})"

    def evaluate(self, state: State) -> float:
        return self.left.evaluate(state) - self.right.evaluate(state)

    def to_clause(self, ctx: JANI) -> tuple[ExprRef, list[ExprRef], list[ExprRef]]:
        left, left_addt, left_vars = self.left.to_clause(ctx)
        right, right_addt, right_vars = self.right.to_clause(ctx)
        return left - right, left_addt + right_addt, left_vars + right_vars


class MulExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left.__repr__()} * {self.right.__repr__()})"

    def evaluate(self, state: State) -> float:
        return self.left.evaluate(state) * self.right.evaluate(state)

    def to_clause(self, ctx: JANI) -> tuple[ExprRef, list[ExprRef], list[ExprRef]]:
        left, left_addt, left_vars = self.left.to_clause(ctx)
        right, right_addt, right_vars = self.right.to_clause(ctx)
        return left * right, left_addt + right_addt, left_vars + right_vars


class DivExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left.__repr__()} / {self.right.__repr__()})"

    def evaluate(self, state: State) -> float:
        return self.left.evaluate(state) / self.right.evaluate(state)

    def to_clause(self, ctx: JANI) -> tuple[ExprRef, list[ExprRef], list[ExprRef]]:
        left, left_addt, left_vars = self.left.to_clause(ctx)
        right, right_addt, right_vars = self.right.to_clause(ctx)
        return left / right, left_addt + right_addt + [right != 0], left_vars + right_vars

class ConjExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left.__repr__()} ∧ {self.right.__repr__()})"

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) and self.right.evaluate(state)

    def to_clause(self, ctx: JANI) -> tuple[ExprRef, list[ExprRef], list[ExprRef]]:
        left, left_addt, left_vars = self.left.to_clause(ctx)
        right, right_addt, right_vars = self.right.to_clause(ctx)
        return And(left, right), left_addt + right_addt, left_vars + right_vars


class DisjExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"(({self.left.__repr__()}) ∨ ({self.right.__repr__()}))"

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) or self.right.evaluate(state)

    def to_clause(self, ctx: JANI) -> tuple[ExprRef, list[ExprRef], list[ExprRef]]:
        left, left_addt, left_vars = self.left.to_clause(ctx)
        right, right_addt, right_vars = self.right.to_clause(ctx)
        return Or(left, right), left_addt + right_addt, left_vars + right_vars


class EqExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left.__repr__()} = {self.right.__repr__()})"

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) == self.right.evaluate(state)

    def to_clause(self, ctx: JANI) -> tuple[ExprRef, list[ExprRef], list[ExprRef]]:
        left, left_addt, left_vars = self.left.to_clause(ctx)
        right, right_addt, right_vars = self.right.to_clause(ctx)
        return left == right, left_addt + right_addt, left_vars + right_vars


class LeExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left.__repr__()} ≤ {self.right.__repr__()})"

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) <= self.right.evaluate(state)

    def to_clause(self, ctx: JANI) -> tuple[ExprRef, list[ExprRef], list[ExprRef]]:
        left, left_addt, left_vars = self.left.to_clause(ctx)
        right, right_addt, right_vars = self.right.to_clause(ctx)
        return left <= right, left_addt + right_addt, left_vars + right_vars


class LtExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left.__repr__()} < {self.right.__repr__()})"

    def evaluate(self, state: State) -> bool:
        return self.left.evaluate(state) < self.right.evaluate(state)

    def to_clause(self, ctx: JANI) -> tuple[ExprRef, list[ExprRef], list[ExprRef]]:
        left, left_addt, left_vars = self.left.to_clause(ctx)
        right, right_addt, right_vars = self.right.to_clause(ctx)
        return left < right, left_addt + right_addt, left_vars + right_vars
