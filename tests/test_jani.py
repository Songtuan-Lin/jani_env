import pytest
import json
import tempfile
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from jani import (
    JANI, Edge, State, Variable, Expression, VarExpression, ConstantExpression,
    AddExpression, SubExpression, MulExpression, ConjExpression, DisjExpression,
    LeExpression, LtExpression, EqExpression, Assignment
)


def create_dummy_goal_failure_files():
    """Create dummy goal and failure files for testing legacy functionality."""
    # Default goal: always true (any state is a goal state)
    goal_data = {
        "goal": {
            "exp": {"left": 0, "op": "≤", "right": 1},
            "op": "state-condition"
        },
        "op": "objective"
    }
    
    # Default failure: never true (no state is a failure state)
    failure_data = {
        "exp": {"left": 1, "op": "≤", "right": 0},
        "op": "state-condition"
    }
    
    goal_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
    failure_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
    
    json.dump(goal_data, goal_file, indent=2)
    json.dump(failure_data, failure_file, indent=2)
    
    goal_file.close()
    failure_file.close()
    
    return goal_file.name, failure_file.name


class TestExpressionEvaluation:
    """Test cases for expression evaluation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test variables
        self.var_x = Variable("x", 0, "int", 5, "bounded", 10, 0)
        self.var_y = Variable("y", 1, "int", 3, "bounded", 10, 0)
        self.var_z = Variable("z", 2, "real", 2.5, "bounded", 10.0, 0.0)
        self.var_flag = Variable("flag", 3, "bool", True, "bounded", None, None)
        
        # Create test state
        self.state = State({
            "x": self.var_x,
            "y": self.var_y, 
            "z": self.var_z,
            "flag": self.var_flag
        })
    
    def test_var_expression_evaluation(self):
        """Test variable expression evaluation."""
        expr = VarExpression("x")
        assert expr.evaluate(self.state) == 5
        
        expr = VarExpression("z")
        assert expr.evaluate(self.state) == 2.5
        
        expr = VarExpression("flag")
        assert expr.evaluate(self.state) == True
    
    def test_arithmetic_expressions(self):
        """Test arithmetic expression evaluation."""
        # Addition
        expr = AddExpression(VarExpression("x"), VarExpression("y"))
        assert expr.evaluate(self.state) == 8
        
        # Subtraction
        expr = SubExpression(VarExpression("x"), VarExpression("y"))
        assert expr.evaluate(self.state) == 2
        
        # Multiplication
        expr = MulExpression(VarExpression("x"), VarExpression("y"))
        assert expr.evaluate(self.state) == 15
    
    def test_boolean_expressions(self):
        """Test boolean expression evaluation."""
        # Less than or equal
        expr = LeExpression(VarExpression("y"), VarExpression("x"))
        assert expr.evaluate(self.state) == True
        
        expr = LeExpression(VarExpression("x"), VarExpression("y"))
        assert expr.evaluate(self.state) == False
        
        # Conjunction
        expr1 = LeExpression(VarExpression("y"), VarExpression("x"))
        expr2 = LeExpression(VarExpression("x"), VarExpression("x"))
        conj_expr = ConjExpression(expr1, expr2)
        assert conj_expr.evaluate(self.state) == True
    
    def test_expression_construct_from_json(self):
        """Test expression construction from JSON."""
        # Simple variable
        json_expr = "x"
        expr = Expression.construct(json_expr)
        assert isinstance(expr, VarExpression)
        assert expr.evaluate(self.state) == 5

        # Addition expression
        json_expr = {
            "op": "+",
            "left": "x",
            "right": "y"
        }
        expr = Expression.construct(json_expr)
        assert isinstance(expr, AddExpression)
        assert expr.evaluate(self.state) == 8

        # Complex nested expression
        json_expr = {
            "op": "≤",
            "left": {
                "op": "+",
                "left": "x",
                "right": "y"
            },
            "right": {
                "op": "*",
                "left": "x",
                "right": "y"
            }
        }
        expr = Expression.construct(json_expr)
        assert expr.evaluate(self.state) == True  # 8 ≤ 15

    def test_constant_expressions(self):
        """Test constant expression evaluation."""
        # Integer constant
        expr = Expression.construct(42)
        assert isinstance(expr, ConstantExpression)
        assert expr.evaluate(self.state) == 42

        # Float constant
        expr = Expression.construct(3.14)
        assert isinstance(expr, ConstantExpression)
        assert expr.evaluate(self.state) == 3.14

        # Boolean constant
        expr = Expression.construct(True)
        assert isinstance(expr, ConstantExpression)
        assert expr.evaluate(self.state) == True

        # Expression with constants
        json_expr = {
            "op": "+",
            "left": "x",
            "right": 10
        }
        expr = Expression.construct(json_expr)
        assert expr.evaluate(self.state) == 15  # 5 + 10

    def test_disjunction_both_true(self):
        """Test disjunction expression where both operands are true."""
        # true OR true = true
        left_expr = ConstantExpression(True)
        right_expr = ConstantExpression(True)
        disj_expr = DisjExpression(left_expr, right_expr)
        assert disj_expr.evaluate(self.state) == True

    def test_disjunction_left_true_right_false(self):
        """Test disjunction expression where left operand is true, right is false."""
        # true OR false = true
        left_expr = ConstantExpression(True)
        right_expr = ConstantExpression(False)
        disj_expr = DisjExpression(left_expr, right_expr)
        assert disj_expr.evaluate(self.state) == True

    def test_disjunction_left_false_right_true(self):
        """Test disjunction expression where left operand is false, right is true."""
        # false OR true = true
        left_expr = ConstantExpression(False)
        right_expr = ConstantExpression(True)
        disj_expr = DisjExpression(left_expr, right_expr)
        assert disj_expr.evaluate(self.state) == True

    def test_disjunction_both_false(self):
        """Test disjunction expression where both operands are false."""
        # false OR false = false
        left_expr = ConstantExpression(False)
        right_expr = ConstantExpression(False)
        disj_expr = DisjExpression(left_expr, right_expr)
        assert disj_expr.evaluate(self.state) == False

    def test_disjunction_with_variables(self):
        """Test disjunction expression with variable expressions."""
        # flag OR (x > 10) = true OR false = true
        left_expr = VarExpression("flag")  # true
        right_expr = LeExpression(ConstantExpression(10), VarExpression("x"))  # 10 <= 5 = false
        disj_expr = DisjExpression(left_expr, right_expr)
        assert disj_expr.evaluate(self.state) == True

        # (x > 10) OR (y > 10) = false OR false = false
        left_expr = LeExpression(ConstantExpression(10), VarExpression("x"))  # 10 <= 5 = false
        right_expr = LeExpression(ConstantExpression(10), VarExpression("y"))  # 10 <= 3 = false
        disj_expr = DisjExpression(left_expr, right_expr)
        assert disj_expr.evaluate(self.state) == False

    def test_disjunction_with_arithmetic_expressions(self):
        """Test disjunction expression with arithmetic sub-expressions."""
        # (x + y <= 10) OR (x * y <= 10) = (8 <= 10) OR (15 <= 10) = true OR false = true
        left_expr = LeExpression(
            AddExpression(VarExpression("x"), VarExpression("y")),
            ConstantExpression(10)
        )
        right_expr = LeExpression(
            MulExpression(VarExpression("x"), VarExpression("y")),
            ConstantExpression(10)
        )
        disj_expr = DisjExpression(left_expr, right_expr)
        assert disj_expr.evaluate(self.state) == True

    def test_disjunction_nested_complex(self):
        """Test complex nested disjunction expressions."""
        # ((x <= y) OR (z > 5)) OR ((flag) OR (x + y > 20))
        # = ((5 <= 3) OR (2.5 > 5)) OR ((true) OR (8 > 20))
        # = (false OR false) OR (true OR false)
        # = false OR true = true
        inner_left = DisjExpression(
            LeExpression(VarExpression("x"), VarExpression("y")),  # 5 <= 3 = false
            LeExpression(ConstantExpression(5), VarExpression("z"))  # 5 <= 2.5 = false
        )
        inner_right = DisjExpression(
            VarExpression("flag"),  # true
            LeExpression(ConstantExpression(20), AddExpression(VarExpression("x"), VarExpression("y")))  # 20 <= 8 = false
        )
        outer_disj = DisjExpression(inner_left, inner_right)
        assert outer_disj.evaluate(self.state) == True

    def test_equality_equal_integers(self):
        """Test equality expression with equal integer values."""
        # x == 5 = true (since x = 5)
        left_expr = VarExpression("x")
        right_expr = ConstantExpression(5)
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == True

        # 3 == y = true (since y = 3)
        left_expr = ConstantExpression(3)
        right_expr = VarExpression("y")
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == True

    def test_equality_unequal_integers(self):
        """Test equality expression with unequal integer values."""
        # x == 10 = false (since x = 5)
        left_expr = VarExpression("x")
        right_expr = ConstantExpression(10)
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == False

        # x == y = false (since x = 5, y = 3)
        left_expr = VarExpression("x")
        right_expr = VarExpression("y")
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == False

    def test_equality_equal_reals(self):
        """Test equality expression with equal real/float values."""
        # z == 2.5 = true (since z = 2.5)
        left_expr = VarExpression("z")
        right_expr = ConstantExpression(2.5)
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == True

        # 2.5 == z = true (since z = 2.5)
        left_expr = ConstantExpression(2.5)
        right_expr = VarExpression("z")
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == True

    def test_equality_unequal_reals(self):
        """Test equality expression with unequal real/float values."""
        # z == 3.0 = false (since z = 2.5)
        left_expr = VarExpression("z")
        right_expr = ConstantExpression(3.0)
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == False

        # z == x = false (since z = 2.5, x = 5)
        left_expr = VarExpression("z")
        right_expr = VarExpression("x")
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == False

    def test_equality_equal_booleans(self):
        """Test equality expression with equal boolean values."""
        # flag == true = true (since flag = true)
        left_expr = VarExpression("flag")
        right_expr = ConstantExpression(True)
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == True

        # true == flag = true (since flag = true)
        left_expr = ConstantExpression(True)
        right_expr = VarExpression("flag")
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == True

    def test_equality_unequal_booleans(self):
        """Test equality expression with unequal boolean values."""
        # flag == false = false (since flag = true)
        left_expr = VarExpression("flag")
        right_expr = ConstantExpression(False)
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == False

        # false == flag = false (since flag = true)
        left_expr = ConstantExpression(False)
        right_expr = VarExpression("flag")
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == False

    def test_equality_with_arithmetic_expressions(self):
        """Test equality expression with arithmetic sub-expressions."""
        # (x + y) == 8 = true (since 5 + 3 = 8)
        left_expr = AddExpression(VarExpression("x"), VarExpression("y"))
        right_expr = ConstantExpression(8)
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == True

        # (x * y) == 15 = true (since 5 * 3 = 15)
        left_expr = MulExpression(VarExpression("x"), VarExpression("y"))
        right_expr = ConstantExpression(15)
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == True

        # (x - y) == 3 = false (since 5 - 3 = 2, not 3)
        left_expr = SubExpression(VarExpression("x"), VarExpression("y"))
        right_expr = ConstantExpression(3)
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == False

    def test_equality_complex_expressions_both_sides(self):
        """Test equality expression with complex expressions on both sides."""
        # (x + y) == (z * 3.2) = 8 == 8.0 = true (since 5 + 3 = 8, 2.5 * 3.2 = 8.0)
        left_expr = AddExpression(VarExpression("x"), VarExpression("y"))
        right_expr = MulExpression(VarExpression("z"), ConstantExpression(3.2))
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == True

        # (x * y) == (x + y + z) = 15 == 10.5 = false (since 5 * 3 = 15, 5 + 3 + 2.5 = 10.5)
        left_expr = MulExpression(VarExpression("x"), VarExpression("y"))
        right_expr = AddExpression(
            AddExpression(VarExpression("x"), VarExpression("y")),
            VarExpression("z")
        )
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == False

    def test_equality_with_boolean_expressions(self):
        """Test equality expression with boolean sub-expressions."""
        # (x <= y) == false = true (since 5 <= 3 is false)
        left_expr = LeExpression(VarExpression("x"), VarExpression("y"))
        right_expr = ConstantExpression(False)
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == True

        # (y <= x) == flag = true (since 3 <= 5 is true, and flag is true)
        left_expr = LeExpression(VarExpression("y"), VarExpression("x"))
        right_expr = VarExpression("flag")
        eq_expr = EqExpression(left_expr, right_expr)
        assert eq_expr.evaluate(self.state) == True

    def test_equality_nested_with_disjunction(self):
        """Test equality expression nested with disjunction expressions."""
        # ((x == 5) OR (y == 5)) == true = true (since x == 5 is true, so the OR is true)
        inner_disj = DisjExpression(
            EqExpression(VarExpression("x"), ConstantExpression(5)),
            EqExpression(VarExpression("y"), ConstantExpression(5))
        )
        eq_expr = EqExpression(inner_disj, ConstantExpression(True))
        assert eq_expr.evaluate(self.state) == True

        # ((x == 10) OR (y == 10)) == false = true (since both comparisons are false, so the OR is false)
        inner_disj = DisjExpression(
            EqExpression(VarExpression("x"), ConstantExpression(10)),
            EqExpression(VarExpression("y"), ConstantExpression(10))
        )
        eq_expr = EqExpression(inner_disj, ConstantExpression(False))
        assert eq_expr.evaluate(self.state) == True

    def test_disjunction_construct_from_json(self):
        """Test disjunction expression construction from JSON."""
        # Simple disjunction: flag ∨ false = true
        json_expr = {
            "op": "∨",
            "left": "flag",
            "right": False
        }
        expr = Expression.construct(json_expr)
        assert isinstance(expr, DisjExpression)
        assert expr.evaluate(self.state) == True

        # Complex disjunction: (x ≤ y) ∨ (x = 5) = false ∨ true = true
        json_expr = {
            "op": "∨",
            "left": {
                "op": "≤",
                "left": "x",
                "right": "y"
            },
            "right": {
                "op": "=",
                "left": "x",
                "right": 5
            }
        }
        expr = Expression.construct(json_expr)
        assert isinstance(expr, DisjExpression)
        assert expr.evaluate(self.state) == True

    def test_equality_construct_from_json(self):
        """Test equality expression construction from JSON."""
        # Simple equality: x = 5 = true
        json_expr = {
            "op": "=",
            "left": "x",
            "right": 5
        }
        expr = Expression.construct(json_expr)
        assert isinstance(expr, EqExpression)
        assert expr.evaluate(self.state) == True

        # Complex equality: (x + y) = 8 = true
        json_expr = {
            "op": "=",
            "left": {
                "op": "+",
                "left": "x",
                "right": "y"
            },
            "right": 8
        }
        expr = Expression.construct(json_expr)
        assert isinstance(expr, EqExpression)
        assert expr.evaluate(self.state) == True

        # Nested equality with disjunction: ((x = 5) ∨ (y = 5)) = true = true
        json_expr = {
            "op": "=",
            "left": {
                "op": "∨",
                "left": {
                    "op": "=",
                    "left": "x",
                    "right": 5
                },
                "right": {
                    "op": "=",
                    "left": "y",
                    "right": 5
                }
            },
            "right": True
        }
        expr = Expression.construct(json_expr)
        assert isinstance(expr, EqExpression)
        assert expr.evaluate(self.state) == True

    def test_mixed_boolean_expressions_comprehensive(self):
        """Test comprehensive combinations of disjunction, equality, and other boolean expressions."""
        # Complex expression: ((x = 5) ∧ (flag)) ∨ ((y ≤ x) = true)
        # = ((true) ∧ (true)) ∨ ((true) = true)
        # = true ∨ true = true
        json_expr = {
            "op": "∨",
            "left": {
                "op": "∧",
                "left": {
                    "op": "=",
                    "left": "x",
                    "right": 5
                },
                "right": "flag"
            },
            "right": {
                "op": "=",
                "left": {
                    "op": "≤",
                    "left": "y",
                    "right": "x"
                },
                "right": True
            }
        }
        expr = Expression.construct(json_expr)
        assert expr.evaluate(self.state) == True

        # Another complex expression: ((x + y) = (z * 3.2)) ∨ ((x * y) = 20)
        # = (8 = 8.0) ∨ (15 = 20)
        # = true ∨ false = true
        json_expr = {
            "op": "∨",
            "left": {
                "op": "=",
                "left": {
                    "op": "+",
                    "left": "x",
                    "right": "y"
                },
                "right": {
                    "op": "*",
                    "left": "z",
                    "right": 3.2
                }
            },
            "right": {
                "op": "=",
                "left": {
                    "op": "*",
                    "left": "x",
                    "right": "y"
                },
                "right": 20
            }
        }
        expr = Expression.construct(json_expr)
        assert expr.evaluate(self.state) == True


class TestGuardEvaluation:
    """Test cases for guard evaluation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test variables
        self.var_x = Variable("x", 0, "int", 5, "bounded", 10, 0)
        self.var_y = Variable("y", 1, "int", 3, "bounded", 10, 0)
        self.var_flag = Variable("flag", 2, "bool", True, "bounded", None, None)
        
        # Create test state
        self.state = State({
            "x": self.var_x,
            "y": self.var_y,
            "flag": self.var_flag
        })
    
    def test_simple_boolean_guard_true(self):
        """Test simple boolean guard that evaluates to true."""
        edge_json = {
            "action": "test_action",
            "guard": {
                "exp": "flag"
            },
            "destinations": [{
                "assignments": [],
                "probability": 1.0
            }]
        }
        
        edge = Edge(edge_json)
        assert edge.is_enabled(self.state) == True
    
    def test_simple_boolean_guard_false(self):
        """Test simple boolean guard that evaluates to false."""
        # Change flag to false
        self.state["flag"] = False
        
        edge_json = {
            "action": "test_action", 
            "guard": {
                "exp": "flag"
            },
            "destinations": [{
                "assignments": [],
                "probability": 1.0
            }]
        }
        
        edge = Edge(edge_json)
        assert edge.is_enabled(self.state) == False
    
    def test_comparison_guard_true(self):
        """Test comparison guard that evaluates to true."""
        edge_json = {
            "action": "test_action",
            "guard": {
                "exp": {
                    "op": "≤",
                    "left": "y",
                    "right": "x"
                }
            },
            "destinations": [{
                "assignments": [],
                "probability": 1.0
            }]
        }
        
        edge = Edge(edge_json)
        assert edge.is_enabled(self.state) == True  # 3 ≤ 5
    
    def test_comparison_guard_false(self):
        """Test comparison guard that evaluates to false."""
        edge_json = {
            "action": "test_action",
            "guard": {
                "exp": {
                    "op": "≤",
                    "left": "x",
                    "right": "y"
                }
            },
            "destinations": [{
                "assignments": [],
                "probability": 1.0
            }]
        }
        
        edge = Edge(edge_json)
        assert edge.is_enabled(self.state) == False  # 5 ≤ 3 is false
    
    def test_complex_guard_with_arithmetic(self):
        """Test complex guard with arithmetic operations."""
        edge_json = {
            "action": "test_action",
            "guard": {
                "exp": {
                    "op": "≤",
                    "left": {
                        "op": "+",
                        "left": "x",
                        "right": "y"
                    },
                    "right": {
                        "op": "*",
                        "left": "x",
                        "right": "y"
                    }
                }
            },
            "destinations": [{
                "assignments": [],
                "probability": 1.0
            }]
        }
        
        edge = Edge(edge_json)
        assert edge.is_enabled(self.state) == True  # (5+3) ≤ (5*3) -> 8 ≤ 15
    
    def test_conjunction_guard_both_true(self):
        """Test conjunction guard where both conditions are true."""
        edge_json = {
            "action": "test_action",
            "guard": {
                "exp": {
                    "op": "∧",
                    "left": {
                        "op": "≤",
                        "left": "y",
                        "right": "x"
                    },
                    "right": "flag"
                }
            },
            "destinations": [{
                "assignments": [],
                "probability": 1.0
            }]
        }
        
        edge = Edge(edge_json)
        assert edge.is_enabled(self.state) == True  # (3 ≤ 5) ∧ true
    
    def test_conjunction_guard_one_false(self):
        """Test conjunction guard where one condition is false."""
        edge_json = {
            "action": "test_action",
            "guard": {
                "exp": {
                    "op": "∧",
                    "left": {
                        "op": "≤",
                        "left": "x",
                        "right": "y"
                    },
                    "right": "flag"
                }
            },
            "destinations": [{
                "assignments": [],
                "probability": 1.0
            }]
        }
        
        edge = Edge(edge_json)
        assert edge.is_enabled(self.state) == False  # (5 ≤ 3) ∧ true -> false ∧ true


class TestStateTransitions:
    """Test cases for state transition functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test variables
        self.var_x = Variable("x", 0, "int", 5, "bounded", 10, 0)
        self.var_y = Variable("y", 1, "int", 3, "bounded", 10, 0)
        self.var_z = Variable("z", 2, "real", 2.5, "bounded", 10.0, 0.0)
        self.var_flag = Variable("flag", 3, "bool", True, "bounded", None, None)

        # Create test state
        self.state = State({
            "x": self.var_x,
            "y": self.var_y,
            "z": self.var_z,
            "flag": self.var_flag
        })

    def test_simple_variable_assignment(self):
        """Test simple variable assignment in state transition."""
        edge_json = {
            "action": "increment_x",
            "guard": {
                "exp": "flag"  # Always true
            },
            "destinations": [{
                "assignments": [{
                    "target": "x",
                    "value": {
                        "op": "+",
                        "left": "x",
                        "right": 1
                    }
                }],
                "probability": 1.0
            }]
        }

        edge = Edge(edge_json)
        new_states, distribution = edge.apply(self.state)

        assert len(new_states) == 1
        assert new_states[0]["x"].value == 6  # 5 + 1
        assert new_states[0]["y"].value == 3  # unchanged
        assert new_states[0]["z"].value == 2.5  # unchanged
        assert new_states[0]["flag"].value == True  # unchanged

    def test_multiple_variable_assignments(self):
        """Test multiple variable assignments in single transition."""
        edge_json = {
            "action": "update_multiple",
            "guard": {
                "exp": "flag"
            },
            "destinations": [{
                "assignments": [
                    {
                        "target": "x",
                        "value": {
                            "op": "*",
                            "left": "x",
                            "right": 2
                        }
                    },
                    {
                        "target": "y",
                        "value": {
                            "op": "+",
                            "left": "y",
                            "right": "x"
                        }
                    },
                    {
                        "target": "flag",
                        "value": "flag"  # Keep same value
                    }
                ],
                "probability": 1.0
            }]
        }

        edge = Edge(edge_json)
        new_states, distribution = edge.apply(self.state)

        assert len(new_states) == 1
        assert new_states[0]["x"].value == 10  # 5 * 2
        assert new_states[0]["y"].value == 8   # 3 + 5 (original x value)
        assert new_states[0]["flag"].value == True

    def test_complex_expression_assignment(self):
        """Test assignment with complex expression."""
        edge_json = {
            "action": "complex_update",
            "guard": {
                "exp": "flag"
            },
            "destinations": [{
                "assignments": [{
                    "target": "z",
                    "value": {
                        "op": "+",
                        "left": {
                            "op": "*",
                            "left": "x",
                            "right": "z"
                        },
                        "right": {
                            "op": "-",
                            "left": "y",
                            "right": 1
                        }
                    }
                }],
                "probability": 1.0
            }]
        }

        edge = Edge(edge_json)
        new_states, distribution = edge.apply(self.state)

        assert len(new_states) == 1
        # z = (x * z) + (y - 1) = (5 * 2.5) + (3 - 1) = 12.5 + 2 = 14.5
        assert new_states[0]["z"].value == 14.5

    def test_guard_false_no_transition(self):
        """Test that no transition occurs when guard is false."""
        # Set flag to false
        self.state["flag"] = False

        edge_json = {
            "action": "blocked_action",
            "guard": {
                "exp": "flag"
            },
            "destinations": [{
                "assignments": [{
                    "target": "x",
                    "value": 999
                }],
                "probability": 1.0
            }]
        }

        edge = Edge(edge_json)
        new_states, distribution = edge.apply(self.state)

        assert len(new_states) == 0  # No transition should occur

    def test_conditional_guard_transition(self):
        """Test transition with conditional guard."""
        edge_json = {
            "action": "conditional_action",
            "guard": {
                "exp": {
                    "op": "≤",
                    "left": "x",
                    "right": 10
                }
            },
            "destinations": [{
                "assignments": [{
                    "target": "x",
                    "value": {
                        "op": "+",
                        "left": "x",
                        "right": 1
                    }
                }],
                "probability": 1.0
            }]
        }

        edge = Edge(edge_json)
        new_states, distribution = edge.apply(self.state)

        assert len(new_states) == 1
        assert new_states[0]["x"].value == 6

    def test_boundary_condition_guard(self):
        """Test guard evaluation at boundary conditions."""
        # Set x to boundary value
        self.state["x"] = 10

        edge_json = {
            "action": "boundary_test",
            "guard": {
                "exp": {
                    "op": "≤",
                    "left": "x",
                    "right": 10
                }
            },
            "destinations": [{
                "assignments": [{
                    "target": "y",
                    "value": 100
                }],
                "probability": 1.0
            }]
        }

        edge = Edge(edge_json)
        assert edge.is_enabled(self.state) == True  # 10 ≤ 10 should be true

        new_states, distribution = edge.apply(self.state)
        assert len(new_states) == 1
        assert new_states[0]["y"].value == 100


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.var_x = Variable("x", 0, "int", 0, "bounded", 100, -100)
        self.var_y = Variable("y", 1, "real", 0.0, "bounded", 100.0, -100.0)
        self.var_flag = Variable("flag", 2, "bool", False, "bounded", None, None)

        self.state = State({
            "x": self.var_x,
            "y": self.var_y,
            "flag": self.var_flag
        })

    def test_zero_values_in_expressions(self):
        """Test expressions with zero values."""
        edge_json = {
            "action": "zero_test",
            "guard": {
                "exp": {
                    "op": "≤",
                    "left": "x",
                    "right": 0
                }
            },
            "destinations": [{
                "assignments": [{
                    "target": "y",
                    "value": {
                        "op": "*",
                        "left": "x",
                        "right": "y"
                    }
                }],
                "probability": 1.0
            }]
        }

        edge = Edge(edge_json)
        assert edge.is_enabled(self.state) == True  # 0 ≤ 0

        new_states, distribution = edge.apply(self.state)
        assert len(new_states) == 1
        assert new_states[0]["y"].value == 0.0  # 0 * 0.0

    def test_negative_values_in_expressions(self):
        """Test expressions with negative values."""
        self.state["x"] = -5
        self.state["y"] = -2.5

        edge_json = {
            "action": "negative_test",
            "guard": {
                "exp": {
                    "op": "≤",
                    "left": "x",
                    "right": 0
                }
            },
            "destinations": [{
                "assignments": [{
                    "target": "x",
                    "value": {
                        "op": "*",
                        "left": "x",
                        "right": -1
                    }
                }],
                "probability": 1.0
            }]
        }

        edge = Edge(edge_json)
        assert edge.is_enabled(self.state) == True  # -5 ≤ 0

        new_states, distribution = edge.apply(self.state)
        assert len(new_states) == 1
        assert new_states[0]["x"].value == 5  # -5 * -1

    def test_boolean_false_in_conjunction(self):
        """Test conjunction with false boolean."""
        edge_json = {
            "action": "false_conj_test",
            "guard": {
                "exp": {
                    "op": "∧",
                    "left": "flag",  # false
                    "right": {
                        "op": "≤",
                        "left": "x",
                        "right": 10
                    }
                }
            },
            "destinations": [{
                "assignments": [{
                    "target": "x",
                    "value": 999
                }],
                "probability": 1.0
            }]
        }

        edge = Edge(edge_json)
        assert edge.is_enabled(self.state) == False  # false ∧ true = false

        new_states, distribution = edge.apply(self.state)
        assert len(new_states) == 0

    def test_state_immutability_during_evaluation(self):
        """Test that original state is not modified during transition."""
        original_x = self.state["x"].value
        original_y = self.state["y"].value

        edge_json = {
            "action": "immutability_test",
            "guard": {
                "exp": {
                    "op": "≤",
                    "left": "x",
                    "right": 10
                }
            },
            "destinations": [{
                "assignments": [
                    {
                        "target": "x",
                        "value": 100
                    },
                    {
                        "target": "y",
                        "value": 200.0
                    }
                ],
                "probability": 1.0
            }]
        }

        edge = Edge(edge_json)
        new_states, distribution = edge.apply(self.state)

        # Original state should be unchanged
        assert self.state["x"].value == original_x
        assert self.state["y"].value == original_y

        # New state should have updated values
        assert len(new_states) == 1
        assert new_states[0]["x"].value == 100
        assert new_states[0]["y"].value == 200.0

    def test_assignment_using_original_state_values(self):
        """Test that assignments use original state values, not intermediate ones."""
        self.state["x"] = 5
        self.state["y"] = 3.0

        edge_json = {
            "action": "original_values_test",
            "guard": {
                "exp": "flag"  # We'll set this to true
            },
            "destinations": [{
                "assignments": [
                    {
                        "target": "x",
                        "value": {
                            "op": "+",
                            "left": "x",
                            "right": "y"  # Should use original y value
                        }
                    },
                    {
                        "target": "y",
                        "value": {
                            "op": "*",
                            "left": "x",  # Should use original x value
                            "right": 2
                        }
                    }
                ],
                "probability": 1.0
            }]
        }

        # Set flag to true for this test
        self.state["flag"] = True

        edge = Edge(edge_json)
        new_states, distribution = edge.apply(self.state)

        assert len(new_states) == 1
        assert new_states[0]["x"].value == 8.0   # 5 + 3 (original y)
        assert new_states[0]["y"].value == 10.0  # 5 * 2 (original x)


class TestJANIIntegration:
    """Integration tests using sample JANI model data."""

    def create_sample_jani_file(self):
        """Create a sample JANI file for testing."""
        jani_data = {
            "jani-version": 1,
            "name": "test-model",
            "type": "lts",
            "actions": [
                {"name": "increment"},
                {"name": "decrement"},
                {"name": "reset"}
            ],
            "constants": [
                {
                    "name": "max_value",
                    "type": "int",
                    "value": 10
                }
            ],
            "variables": [
                {
                    "name": "counter",
                    "type": {
                        "base": "int",
                        "kind": "bounded",
                        "lower-bound": 0,
                        "upper-bound": 20
                    },
                    "initial-value": 5
                },
                {
                    "name": "active",
                    "type": {
                        "base": "bool",
                        "kind": "bounded"
                    },
                    "initial-value": True
                }
            ],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["init"],
                "locations": [{"name": "init"}],
                "edges": [
                    {
                        "action": "increment",
                        "guard": {
                            "exp": {
                                "op": "∧",
                                "left": "active",
                                "right": {
                                    "op": "≤",
                                    "left": "counter",
                                    "right": "max_value"
                                }
                            }
                        },
                        "destinations": [{
                            "assignments": [{
                                "target": "counter",
                                "value": {
                                    "op": "+",
                                    "left": "counter",
                                    "right": 1
                                }
                            }],
                            "probability": 1.0
                        }]
                    },
                    {
                        "action": "decrement",
                        "guard": {
                            "exp": {
                                "op": "∧",
                                "left": "active",
                                "right": {
                                    "op": "≤",
                                    "left": 1,
                                    "right": "counter"
                                }
                            }
                        },
                        "destinations": [{
                            "assignments": [{
                                "target": "counter",
                                "value": {
                                    "op": "-",
                                    "left": "counter",
                                    "right": 1
                                }
                            }],
                            "probability": 1.0
                        }]
                    },
                    {
                        "action": "reset",
                        "guard": {
                            "exp": "active"
                        },
                        "destinations": [{
                            "assignments": [
                                {
                                    "target": "counter",
                                    "value": 0
                                },
                                {
                                    "target": "active",
                                    "value": "active"  # Keep same value
                                }
                            ],
                            "probability": 1.0
                        }]
                    }
                ]
            }]
        }

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(jani_data, temp_file, indent=2)
        temp_file.close()
        return temp_file.name

    def create_minimal_start_file_for_integration(self):
        """Create a minimal start file for integration testing."""
        start_data = {
            "op": "states-values",
            "values": [
                {
                    "variables": [
                        {"var": "counter", "value": 5},
                        {"var": "active", "value": True}
                    ]
                }
            ]
        }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(start_data, temp_file, indent=2)
        temp_file.close()
        return temp_file.name

    def test_jani_model_loading_and_edge_evaluation(self):
        """Test loading JANI model and evaluating edges."""
        jani_file = self.create_sample_jani_file()
        start_file = self.create_minimal_start_file_for_integration()

        try:
            # Load JANI model
            goal_file, failure_file = create_dummy_goal_failure_files()
            goal_file, failure_file = create_dummy_goal_failure_files()

            jani = JANI(jani_file, start_file, goal_file, failure_file)

            # Create initial state
            state_vars = {}
            for var in jani._variables:
                state_vars[var.name] = var
            for const in jani._constants:
                state_vars[const.name] = const

            initial_state = State(state_vars)

            # Test increment edge (should be enabled)
            increment_edge = jani._automata[0]._edges["increment"][0]  # First increment edge
            assert increment_edge.is_enabled(initial_state) == True

            # Apply increment transition
            new_states, distribution = increment_edge.apply(initial_state)
            assert len(new_states) == 1
            assert new_states[0]["counter"].value == 6  # 5 + 1

            # Test decrement edge (should be enabled)
            decrement_edge = jani._automata[0]._edges["decrement"][0]  # First decrement edge
            assert decrement_edge.is_enabled(initial_state) == True

            # Apply decrement transition
            new_states, distribution = decrement_edge.apply(initial_state)
            assert len(new_states) == 1
            assert new_states[0]["counter"].value == 4  # 5 - 1

            # Test reset edge
            reset_edge = jani._automata[0]._edges["reset"][0]  # First reset edge
            assert reset_edge.is_enabled(initial_state) == True

            new_states, distribution = reset_edge.apply(initial_state)
            assert len(new_states) == 1
            assert new_states[0]["counter"].value == 0
            assert new_states[0]["active"].value == True

        finally:
            # Clean up temporary files
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()

    def test_boundary_conditions_with_jani_model(self):
        """Test boundary conditions using JANI model."""
        jani_file = self.create_sample_jani_file()

        try:
            start_file = self.create_minimal_start_file_for_integration()
            goal_file, failure_file = create_dummy_goal_failure_files()
            goal_file, failure_file = create_dummy_goal_failure_files()

            jani = JANI(jani_file, start_file, goal_file, failure_file)

            # Create state at upper boundary
            state_vars = {}
            for var in jani._variables:
                state_vars[var.name] = var
            for const in jani._constants:
                state_vars[const.name] = const

            boundary_state = State(state_vars)
            boundary_state["counter"] = 10  # At max_value

            # Increment should still be enabled (counter ≤ max_value)
            increment_edge = jani._automata[0]._edges["increment"][0]
            assert increment_edge.is_enabled(boundary_state) == True

            # Apply increment
            new_states, distribution = increment_edge.apply(boundary_state)
            assert len(new_states) == 1
            assert new_states[0]["counter"].value == 11

            # Now test at counter = 11 (above max_value)
            above_boundary_state = State(state_vars)
            above_boundary_state["counter"] = 11

            # Increment should be disabled (11 ≤ 10 is false)
            assert increment_edge.is_enabled(above_boundary_state) == False

            # No transition should occur
            new_states, distribution = increment_edge.apply(above_boundary_state)
            assert len(new_states) == 0

        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()


class TestErrorHandling:
    """Test cases for error handling and invalid inputs."""

    def test_unsupported_operator_in_expression(self):
        """Test error handling for unsupported operators."""
        with pytest.raises(ValueError, match="Unsupported operator"):
            Expression.construct({
                "op": "unsupported_op",
                "left": "x",
                "right": "y"
            })

    def test_missing_variable_in_state(self):
        """Test error when referencing non-existent variable."""
        var_x = Variable("x", 0, "int", 5, "bounded", 10, 0)
        state = State({"x": var_x})

        expr = VarExpression("nonexistent_var")

        with pytest.raises(KeyError):
            expr.evaluate(state)


class TestWithSimpleJANIFile:
    """Test cases using the simple test JANI file."""

    def test_simple_jani_file_loading(self):
        """Test loading and basic operations with simple JANI file."""
        jani_file = "../examples/simple_test.jani"

        # Check if file exists, skip test if not
        if not Path(jani_file).exists():
            pytest.skip(f"Test JANI file {jani_file} not found")
        
        # Need to create a start file for the simple test
        start_data = {
            "op": "states-values",
            "values": [
                {
                    "variables": [
                        {"var": "count", "value": 10},
                        {"var": "value", "value": 5.0},
                        {"var": "enabled", "value": True}
                    ]
                }
            ]
        }
        
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(start_data, start_file, indent=2)
        start_file.close()
        
        try:
            goal_file, failure_file = create_dummy_goal_failure_files()

            jani = JANI(jani_file, start_file.name, goal_file, failure_file)

            # Verify model loaded correctly
            assert len(jani._actions) == 3
            assert len(jani._constants) == 2
            assert len(jani._variables) == 3
            assert len(jani._automata) == 1

            # Create initial state
            state_vars = {}
            for var in jani._variables:
                state_vars[var.name] = var
            for const in jani._constants:
                state_vars[const.name] = const

            initial_state = State(state_vars)

            # Test increment edge
            increment_edge = jani._automata[0]._edges["increment"][0]
            assert increment_edge.is_enabled(initial_state) == True

            new_states, distribution = increment_edge.apply(initial_state)
            assert len(new_states) == 1
            assert new_states[0]["count"].value == 12  # 10 + 2 (step_size)

            # Test multiply edge
            multiply_edge = jani._automata[0]._edges["multiply"][0]
            assert multiply_edge.is_enabled(initial_state) == True

            new_states, distribution = multiply_edge.apply(initial_state)
            assert len(new_states) == 1
            assert new_states[0]["value"].value == 7.5  # 5.0 * 1.5
            assert new_states[0]["count"].value == 9    # 10 - 1
        
        finally:
            Path(start_file.name).unlink()

    def test_complex_guard_evaluation_with_constants(self):
        """Test guard evaluation involving constants."""
        jani_file = "../examples/simple_test.jani"

        if not Path(jani_file).exists():
            pytest.skip(f"Test JANI file {jani_file} not found")
        
        # Need to create a start file for the simple test
        start_data = {
            "op": "states-values",
            "values": [
                {
                    "variables": [
                        {"var": "count", "value": 10},
                        {"var": "value", "value": 5.0},
                        {"var": "enabled", "value": True}
                    ]
                }
            ]
        }
        
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(start_data, start_file, indent=2)
        start_file.close()
        
        try:
            goal_file, failure_file = create_dummy_goal_failure_files()

            jani = JANI(jani_file, start_file.name, goal_file, failure_file)

            # Create state at boundary
            state_vars = {}
            for var in jani._variables:
                state_vars[var.name] = var
            for const in jani._constants:
                state_vars[const.name] = const

            boundary_state = State(state_vars)
            boundary_state["count"] = 48  # 48 + 2 = 50 (at boundary)

            # Increment should be enabled (48 + 2 ≤ 50)
            increment_edge = jani._automata[0]._edges["increment"][0]
            assert increment_edge.is_enabled(boundary_state) == True

            # Set count to 49 (49 + 2 = 51 > 50)
            boundary_state["count"] = 49
            assert increment_edge.is_enabled(boundary_state) == False
        
        finally:
            Path(start_file.name).unlink()


class TestJANIFileLoading:
    """Test cases for JANI file loading functionality."""

    def create_minimal_jani_file(self):
        """Create a minimal JANI file for testing."""
        jani_data = {
            "jani-version": 1,
            "name": "minimal-test",
            "type": "lts",
            "actions": [
                {"name": "action1"}
            ],
            "constants": [
                {
                    "name": "const1",
                    "type": "int",
                    "value": 42
                }
            ],
            "variables": [
                {
                    "name": "var1",
                    "type": {
                        "base": "int",
                        "kind": "bounded",
                        "lower-bound": 0,
                        "upper-bound": 100
                    },
                    "initial-value": 10
                }
            ],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": [{
                    "action": "action1",
                    "guard": {"exp": True},
                    "destinations": [{
                        "assignments": [],
                        "probability": 1.0
                    }]
                }]
            }]
        }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(jani_data, temp_file, indent=2)
        temp_file.close()
        return temp_file.name

    def create_minimal_start_file(self):
        """Create a minimal start file for testing."""
        start_data = {
            "op": "states-values",
            "values": [
                {
                    "variables": [
                        {"var": "var1", "value": 5}
                    ]
                },
                {
                    "variables": [
                        {"var": "var1", "value": 15}
                    ]
                },
                {
                    "variables": [
                        {"var": "var1", "value": 25}
                    ]
                }
            ]
        }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(start_data, temp_file, indent=2)
        temp_file.close()
        return temp_file.name

    def test_jani_file_loading_basic(self):
        """Test basic JANI file loading."""
        jani_file = self.create_minimal_jani_file()
        start_file = self.create_minimal_start_file()
        
        try:
            goal_file, failure_file = create_dummy_goal_failure_files()

            jani = JANI(jani_file, start_file, goal_file, failure_file)
            
            # Test that model loaded correctly
            assert len(jani._actions) == 1
            assert jani._actions[0].label == "action1"
            assert len(jani._constants) == 1
            assert jani._constants[0].name == "const1"
            assert jani._constants[0].value == 42
            assert len(jani._variables) == 1
            assert jani._variables[0].name == "var1"
            assert jani._variables[0].value == 10
            assert len(jani._automata) == 1
            assert jani._automata[0]._name == "test_automaton"
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()

    def test_jani_constants_loading(self):
        """Test loading of constants with different types."""
        jani_data = {
            "jani-version": 1,
            "name": "constants-test",
            "type": "lts",
            "actions": [{"name": "test"}],
            "constants": [
                {"name": "int_const", "type": "int", "value": 123},
                {"name": "real_const", "type": "real", "value": 3.14},
                {"name": "bool_const", "type": "bool", "value": "true"}
            ],
            "variables": [],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {"op": "states-values", "values": []}
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(jani_data, jani_file, indent=2)
        jani_file.close()
        
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(start_data, start_file, indent=2)
        start_file.close()
        
        try:
            goal_file, failure_file = create_dummy_goal_failure_files()

            jani = JANI(jani_file.name, start_file.name, goal_file, failure_file)
            
            assert len(jani._constants) == 3
            
            # Find constants by name
            int_const = next(c for c in jani._constants if c.name == "int_const")
            real_const = next(c for c in jani._constants if c.name == "real_const")
            bool_const = next(c for c in jani._constants if c.name == "bool_const")
            
            assert int_const.value == 123
            assert int_const.type == "int"
            assert real_const.value == 3.14
            assert real_const.type == "real"
            assert bool_const.value == True
            assert bool_const.type == "bool"
            
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()

    def test_jani_variables_loading(self):
        """Test loading of variables with different types and bounds."""
        jani_data = {
            "jani-version": 1,
            "name": "variables-test",
            "type": "lts",
            "actions": [{"name": "test"}],
            "constants": [],
            "variables": [
                {
                    "name": "int_var",
                    "type": {"base": "int", "kind": "bounded", "lower-bound": -10, "upper-bound": 50},
                    "initial-value": 25
                },
                {
                    "name": "real_var",
                    "type": {"base": "real", "kind": "bounded", "lower-bound": 0.0, "upper-bound": 100.0},
                    "initial-value": 42.5
                },
                {
                    "name": "bool_var",
                    "type": {"base": "bool", "kind": "bounded"},
                    "initial-value": False
                }
            ],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {"op": "states-values", "values": []}
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(jani_data, jani_file, indent=2)
        jani_file.close()
        
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(start_data, start_file, indent=2)
        start_file.close()
        
        try:
            goal_file, failure_file = create_dummy_goal_failure_files()

            jani = JANI(jani_file.name, start_file.name, goal_file, failure_file)
            
            assert len(jani._variables) == 3
            
            # Find variables by name
            int_var = next(v for v in jani._variables if v.name == "int_var")
            real_var = next(v for v in jani._variables if v.name == "real_var")
            bool_var = next(v for v in jani._variables if v.name == "bool_var")
            
            assert int_var.value == 25
            assert int_var.type == "int"
            assert int_var.lower_bound == -10
            assert int_var.upper_bound == 50
            
            assert real_var.value == 42.5
            assert real_var.type == "real"
            assert real_var.lower_bound == 0.0
            assert real_var.upper_bound == 100.0
            
            assert bool_var.value == False
            assert bool_var.type == "bool"
            assert bool_var.lower_bound is None
            assert bool_var.upper_bound is None
            
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()

    def test_multiple_automata_error(self):
        """Test that multiple automata raise an error."""
        jani_data = {
            "jani-version": 1,
            "name": "multi-automata",
            "type": "lts",
            "actions": [{"name": "test"}],
            "constants": [],
            "variables": [],
            "automata": [
                {
                    "name": "automaton1",
                    "initial-locations": ["loc1"],
                    "locations": [{"name": "loc1"}],
                    "edges": []
                },
                {
                    "name": "automaton2",
                    "initial-locations": ["loc1"],
                    "locations": [{"name": "loc1"}],
                    "edges": []
                }
            ]
        }
        
        start_data = {"op": "states-values", "values": []}
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(jani_data, jani_file, indent=2)
        jani_file.close()
        
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(start_data, start_file, indent=2)
        start_file.close()
        
        try:
            goal_file, failure_file = create_dummy_goal_failure_files()
            with pytest.raises(ValueError, match="Multiple automata are not supported yet"):
                JANI(jani_file.name, start_file.name, goal_file, failure_file)
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()


class TestResetMethod:
    """Test cases for the reset method and initial state generation."""

    def create_test_jani_with_start_states(self):
        """Create test JANI and start files for reset testing."""
        jani_data = {
            "jani-version": 1,
            "name": "reset-test",
            "type": "lts",
            "actions": [{"name": "test_action"}],
            "constants": [
                {"name": "const1", "type": "int", "value": 100}
            ],
            "variables": [
                {
                    "name": "x",
                    "type": {"base": "int", "kind": "bounded", "lower-bound": 0, "upper-bound": 100},
                    "initial-value": 10
                },
                {
                    "name": "y",
                    "type": {"base": "real", "kind": "bounded", "lower-bound": 0.0, "upper-bound": 50.0},
                    "initial-value": 5.0
                }
            ],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {
            "op": "states-values",
            "values": [
                {
                    "variables": [
                        {"var": "x", "value": 1},
                        {"var": "y", "value": 1.5}
                    ]
                },
                {
                    "variables": [
                        {"var": "x", "value": 2},
                        {"var": "y", "value": 2.5}
                    ]
                },
                {
                    "variables": [
                        {"var": "x", "value": 3},
                        {"var": "y", "value": 3.5}
                    ]
                },
                {
                    "variables": [
                        {"var": "x", "value": 4},
                        {"var": "y", "value": 4.5}
                    ]
                },
                {
                    "variables": [
                        {"var": "x", "value": 5},
                        {"var": "y", "value": 5.5}
                    ]
                }
            ]
        }
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(jani_data, jani_file, indent=2)
        jani_file.close()
        
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(start_data, start_file, indent=2)
        start_file.close()
        
        return jani_file.name, start_file.name

    def test_reset_returns_valid_state(self):
        """Test that reset method returns a valid state from the pool."""
        jani_file, start_file = self.create_test_jani_with_start_states()
        
        try:
            goal_file, failure_file = create_dummy_goal_failure_files()

            jani = JANI(jani_file, start_file, goal_file, failure_file)
            
            # Test multiple resets to ensure they all return valid states
            expected_x_values = {1, 2, 3, 4, 5}
            expected_y_values = {1.5, 2.5, 3.5, 4.5, 5.5}
            
            for _ in range(20):  # Test multiple times
                state = jani.reset()
                
                # Verify state structure
                assert isinstance(state, State)
                assert "x" in state.variable_dict
                assert "y" in state.variable_dict
                assert "const1" in state.variable_dict
                
                # Verify values are from the expected pool
                assert state["x"].value in expected_x_values
                assert state["y"].value in expected_y_values
                assert state["const1"].value == 100
                
                # Verify corresponding x and y values
                x_val = state["x"].value
                y_val = state["y"].value
                expected_y = x_val + 0.5
                assert y_val == expected_y
                
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()

    def test_reset_randomness(self):
        """Test that reset method actually selects randomly from the pool."""
        jani_file, start_file = self.create_test_jani_with_start_states()
        
        try:
            goal_file, failure_file = create_dummy_goal_failure_files()

            jani = JANI(jani_file, start_file, goal_file, failure_file)
            
            # Collect results from many resets
            results = []
            for _ in range(100):
                state = jani.reset()
                results.append(state["x"].value)
            
            # Check that we got variety in results (not just one value)
            unique_results = set(results)
            assert len(unique_results) > 1, "Reset should return different values over multiple calls"
            
            # Check that all unique results are valid
            expected_values = {1, 2, 3, 4, 5}
            assert unique_results.issubset(expected_values)
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()

    def test_reset_state_independence(self):
        """Test that states returned by reset are independent (deep copied)."""
        jani_file, start_file = self.create_test_jani_with_start_states()
        
        try:
            goal_file, failure_file = create_dummy_goal_failure_files()

            jani = JANI(jani_file, start_file, goal_file, failure_file)
            
            # Get two states
            state1 = jani.reset()
            state2 = jani.reset()
            
            # Modify first state
            original_x1 = state1["x"].value
            state1["x"] = 999
            
            # Verify second state is unaffected
            assert state2["x"].value != 999
            
            # Get another state and verify it's also unaffected
            state3 = jani.reset()
            assert state3["x"].value != 999
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()

    def test_fixed_generator_state_creation(self):
        """Test the FixedGenerator state creation directly."""
        jani_file, start_file = self.create_test_jani_with_start_states()
        
        try:
            goal_file, failure_file = create_dummy_goal_failure_files()

            jani = JANI(jani_file, start_file, goal_file, failure_file)
            
            # Test that the init generator was created correctly
            assert hasattr(jani, '_init_generator')
            assert isinstance(jani._init_generator, jani.FixedGenerator)
            
            # Test that the pool has the correct size
            assert len(jani._init_generator._pool) == 5
            
            # Test that each state in the pool is valid
            for state in jani._init_generator._pool:
                assert isinstance(state, State)
                assert "x" in state.variable_dict
                assert "y" in state.variable_dict
                assert "const1" in state.variable_dict
                
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()

    def test_unsupported_init_generator_type(self):
        """Test error handling for unsupported init generator types."""
        jani_data = {
            "jani-version": 1,
            "name": "unsupported-test",
            "type": "lts",
            "actions": [{"name": "test"}],
            "constants": [],
            "variables": [],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {
            "op": "unsupported-operation",
            "values": []
        }
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(jani_data, jani_file, indent=2)
        jani_file.close()
        
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        json.dump(start_data, start_file, indent=2)
        start_file.close()
        
        try:
            goal_file, failure_file = create_dummy_goal_failure_files()
            with pytest.raises(ValueError, match="Unsupported init state generator operation"):
                JANI(jani_file.name, start_file.name, goal_file, failure_file)
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()


class TestBouncingBallIntegration:
    """Integration tests using the actual bouncing_ball.jani and start.jani files."""

    def test_bouncing_ball_file_loading(self):
        """Test loading the actual bouncing ball JANI file."""
        jani_file = "../examples/bouncing_ball/bouncing_ball.jani"
        start_file = "../examples/bouncing_ball/start.jani"
        
        if not Path(jani_file).exists() or not Path(start_file).exists():
            pytest.skip(f"Required files {jani_file} or {start_file} not found")
        
        # This should not raise an exception
        goal_file, failure_file = create_dummy_goal_failure_files()

        jani = JANI(jani_file, start_file, goal_file, failure_file)
        
        # Verify basic structure
        assert len(jani._actions) >= 1
        assert len(jani._constants) >= 1
        assert len(jani._variables) >= 1
        assert len(jani._automata) == 1
        
        # Verify specific bouncing ball variables are present
        variable_names = {var.name for var in jani._variables}
        assert "height" in variable_names
        assert "velocity" in variable_names
        assert "episode" in variable_names
        
        # Verify actions
        action_names = {action.label for action in jani._actions}
        assert "push" in action_names or "skip" in action_names

    def test_bouncing_ball_reset_functionality(self):
        """Test reset functionality with actual bouncing ball files."""
        jani_file = "../examples/bouncing_ball/bouncing_ball.jani"
        start_file = "../examples/bouncing_ball/start.jani"
        
        if not Path(jani_file).exists() or not Path(start_file).exists():
            pytest.skip(f"Required files {jani_file} or {start_file} not found")
        
        goal_file, failure_file = create_dummy_goal_failure_files()

        
        jani = JANI(jani_file, start_file, goal_file, failure_file)
        
        # Test that reset works and returns valid states
        for _ in range(10):
            state = jani.reset()
            
            # Verify state structure
            assert isinstance(state, State)
            assert "height" in state.variable_dict
            assert "velocity" in state.variable_dict
            assert "episode" in state.variable_dict
            
            # Verify types and bounds are reasonable for bouncing ball
            height = state["height"].value
            velocity = state["velocity"].value
            episode = state["episode"].value
            
            assert isinstance(height, (int, float))
            assert isinstance(velocity, (int, float))
            assert isinstance(episode, int)
            
            # Basic sanity checks for bouncing ball physics
            assert height >= 0  # Height should be non-negative
            assert episode >= 0  # Episode should be non-negative

    def test_bouncing_ball_state_diversity(self):
        """Test that reset returns diverse states from the bouncing ball model."""
        jani_file = "../examples/bouncing_ball/bouncing_ball.jani"
        start_file = "../examples/bouncing_ball/start.jani"
        
        if not Path(jani_file).exists() or not Path(start_file).exists():
            pytest.skip(f"Required files {jani_file} or {start_file} not found")
        
        goal_file, failure_file = create_dummy_goal_failure_files()

        
        jani = JANI(jani_file, start_file, goal_file, failure_file)
        
        # Collect initial states
        heights = []
        velocities = []
        
        for _ in range(50):
            state = jani.reset()
            heights.append(state["height"].value)
            velocities.append(state["velocity"].value)
        
        # Check for diversity
        unique_heights = set(heights)
        unique_velocities = set(velocities)
        
        # Should have multiple different initial states
        assert len(unique_heights) > 1, "Should have diverse height values"
        assert len(unique_velocities) > 1, "Should have diverse velocity values"

    def test_bouncing_ball_automaton_integration(self):
        """Test that automaton can be used with reset states."""
        jani_file = "../examples/bouncing_ball/bouncing_ball.jani"
        start_file = "../examples/bouncing_ball/start.jani"
        
        if not Path(jani_file).exists() or not Path(start_file).exists():
            pytest.skip(f"Required files {jani_file} or {start_file} not found")
        
        goal_file, failure_file = create_dummy_goal_failure_files()

        
        jani = JANI(jani_file, start_file, goal_file, failure_file)
        
        # Get an initial state
        state = jani.reset()
        
        # Test that we can use the automaton with this state
        automaton = jani._automata[0]
        
        # Check that some actions have edges
        assert len(automaton._edges) > 0
        
        # Try to find an enabled action
        found_enabled_action = False
        for action_label, edges in automaton._edges.items():
            for edge in edges:
                if edge.is_enabled(state):
                    found_enabled_action = True
                    
                    # Try to apply the edge
                    new_states, distribution = edge.apply(state)
                    
                    # Should get some result if enabled
                    if len(new_states) > 0:
                        # Verify new state structure
                        new_state = new_states[0]
                        assert isinstance(new_state, State)
                        assert "height" in new_state.variable_dict
                        assert "velocity" in new_state.variable_dict
                        assert "episode" in new_state.variable_dict
                        break
            if found_enabled_action:
                break
        
        # Note: We don't assert found_enabled_action because initial states 
        # might not have any enabled transitions


class TestGoalAndFailureConditions:
    """Test cases for goal and failure condition functionality."""
    
    def test_simple_goal_expression_construction(self):
        """Test construction of simple goal expressions from JSON."""
        # Test simple comparison goal (episode >= 1000)
        goal_json = {
            "goal": {
                "exp": {
                    "left": 1000,
                    "op": "≤",
                    "right": "episode"
                },
                "op": "state-condition"
            },
            "op": "objective"
        }
        
        # Create temp files for testing
        jani_data = {
            "jani-version": 1,
            "name": "test-goal",
            "type": "lts",
            "actions": [{"name": "increment"}],
            "constants": [],
            "variables": [{
                "name": "episode",
                "type": {"kind": "bounded", "base": "int", "lower-bound": 0, "upper-bound": 2000},
                "initial-value": 0
            }],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {
            "op": "states-values",
            "values": [{
                "variables": [{"var": "episode", "value": 500}]
            }]
        }
        
        failure_data = {
            "exp": {"left": "episode", "op": "≤", "right": -1},
            "op": "state-condition"
        }
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        goal_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        failure_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        
        json.dump(jani_data, jani_file, indent=2)
        json.dump(start_data, start_file, indent=2)
        json.dump(goal_json, goal_file, indent=2)
        json.dump(failure_data, failure_file, indent=2)
        
        jani_file.close()
        start_file.close()
        goal_file.close()
        failure_file.close()
        
        try:
            jani = JANI(jani_file.name, start_file.name, goal_file.name, failure_file.name)
            
            # Verify goal expression was constructed
            assert jani._goal_expr is not None
            assert hasattr(jani._goal_expr, 'evaluate')
            
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file.name).unlink()
            Path(failure_file.name).unlink()

    def test_simple_failure_expression_construction(self):
        """Test construction of simple failure expressions from JSON."""
        # Test simple failure condition
        failure_json = {
            "exp": {
                "left": "height",
                "op": "≤",
                "right": 0
            },
            "op": "state-condition"
        }
        
        jani_data = {
            "jani-version": 1,
            "name": "test-failure",
            "type": "lts",
            "actions": [{"name": "fall"}],
            "constants": [],
            "variables": [{
                "name": "height",
                "type": {"kind": "bounded", "base": "real", "lower-bound": 0.0, "upper-bound": 100.0},
                "initial-value": 50.0
            }],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {
            "op": "states-values",
            "values": [{
                "variables": [{"var": "height", "value": 25.0}]
            }]
        }
        
        goal_data = {
            "goal": {
                "exp": {"left": "height", "op": "≤", "right": 100},
                "op": "state-condition"
            },
            "op": "objective"
        }
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        goal_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        failure_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        
        json.dump(jani_data, jani_file, indent=2)
        json.dump(start_data, start_file, indent=2)
        json.dump(goal_data, goal_file, indent=2)
        json.dump(failure_json, failure_file, indent=2)
        
        jani_file.close()
        start_file.close()
        goal_file.close()
        failure_file.close()
        
        try:
            jani = JANI(jani_file.name, start_file.name, goal_file.name, failure_file.name)
            
            # Verify failure expression was constructed
            assert jani._failure_expr is not None
            assert hasattr(jani._failure_expr, 'evaluate')
            
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file.name).unlink()
            Path(failure_file.name).unlink()

    def test_complex_goal_expression_construction(self):
        """Test construction of complex goal expressions with nested operations."""
        # Test complex arithmetic and boolean goal condition
        goal_json = {
            "goal": {
                "exp": {
                    "left": {
                        "left": "x",
                        "op": "+",
                        "right": "y"
                    },
                    "op": "≤",
                    "right": {
                        "left": "z",
                        "op": "*",
                        "right": 2
                    }
                },
                "op": "state-condition"
            },
            "op": "objective"
        }
        
        jani_data = {
            "jani-version": 1,
            "name": "test-complex-goal",
            "type": "lts",
            "actions": [{"name": "test"}],
            "constants": [],
            "variables": [
                {"name": "x", "type": {"kind": "bounded", "base": "int", "lower-bound": 0, "upper-bound": 50}, "initial-value": 10},
                {"name": "y", "type": {"kind": "bounded", "base": "int", "lower-bound": 0, "upper-bound": 50}, "initial-value": 20},
                {"name": "z", "type": {"kind": "bounded", "base": "int", "lower-bound": 0, "upper-bound": 50}, "initial-value": 30}
            ],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {
            "op": "states-values",
            "values": [{
                "variables": [
                    {"var": "x", "value": 10},
                    {"var": "y", "value": 20},
                    {"var": "z", "value": 30}
                ]
            }]
        }
        
        failure_data = {
            "exp": {"left": "x", "op": "≤", "right": -1},
            "op": "state-condition"
        }
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        goal_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        failure_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        
        json.dump(jani_data, jani_file, indent=2)
        json.dump(start_data, start_file, indent=2)
        json.dump(goal_json, goal_file, indent=2)
        json.dump(failure_data, failure_file, indent=2)
        
        jani_file.close()
        start_file.close()
        goal_file.close()
        failure_file.close()
        
        try:
            jani = JANI(jani_file.name, start_file.name, goal_file.name, failure_file.name)
            
            # Verify complex goal expression was constructed
            assert jani._goal_expr is not None
            
            # Test expression structure by evaluating
            state = jani.reset()
            result = jani._goal_expr.evaluate(state)
            assert isinstance(result, bool)
            
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file.name).unlink()
            Path(failure_file.name).unlink()

    def test_goal_state_evaluation(self):
        """Test evaluation of states against goal conditions."""
        # Create a simple goal: episode >= 1000
        goal_json = {
            "goal": {
                "exp": {
                    "left": 1000,
                    "op": "≤",
                    "right": "episode"
                },
                "op": "state-condition"
            },
            "op": "objective"
        }
        
        jani_data = {
            "jani-version": 1,
            "name": "test-goal-eval",
            "type": "lts",
            "actions": [{"name": "increment"}],
            "constants": [],
            "variables": [{
                "name": "episode",
                "type": {"kind": "bounded", "base": "int", "lower-bound": 0, "upper-bound": 2000},
                "initial-value": 0
            }],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {
            "op": "states-values",
            "values": [
                {"variables": [{"var": "episode", "value": 999}]},
                {"variables": [{"var": "episode", "value": 1000}]},
                {"variables": [{"var": "episode", "value": 1001}]}
            ]
        }
        
        failure_data = {
            "exp": {"left": "episode", "op": "≤", "right": -1},
            "op": "state-condition"
        }
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        goal_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        failure_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        
        json.dump(jani_data, jani_file, indent=2)
        json.dump(start_data, start_file, indent=2)
        json.dump(goal_json, goal_file, indent=2)
        json.dump(failure_data, failure_file, indent=2)
        
        jani_file.close()
        start_file.close()
        goal_file.close()
        failure_file.close()
        
        try:
            jani = JANI(jani_file.name, start_file.name, goal_file.name, failure_file.name)
            
            # Create test states manually
            var_episode_999 = Variable("episode", 0, "int", 999, "bounded", 2000, 0)
            var_episode_1000 = Variable("episode", 0, "int", 1000, "bounded", 2000, 0)
            var_episode_1001 = Variable("episode", 0, "int", 1001, "bounded", 2000, 0)
            
            state_999 = State({"episode": var_episode_999})
            state_1000 = State({"episode": var_episode_1000})
            state_1001 = State({"episode": var_episode_1001})
            
            # Test goal evaluation
            assert not jani.goal_reached(state_999)  # 999 < 1000, so goal not reached
            assert jani.goal_reached(state_1000)     # 1000 >= 1000, so goal reached
            assert jani.goal_reached(state_1001)     # 1001 >= 1000, so goal reached
            
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file.name).unlink()
            Path(failure_file.name).unlink()

    def test_failure_state_evaluation(self):
        """Test evaluation of states against failure conditions."""
        # Create a simple failure condition: height <= 0
        failure_json = {
            "exp": {
                "left": "height",
                "op": "≤",
                "right": 0
            },
            "op": "state-condition"
        }
        
        jani_data = {
            "jani-version": 1,
            "name": "test-failure-eval",
            "type": "lts",
            "actions": [{"name": "fall"}],
            "constants": [],
            "variables": [{
                "name": "height",
                "type": {"kind": "bounded", "base": "real", "lower-bound": -10.0, "upper-bound": 100.0},
                "initial-value": 50.0
            }],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {
            "op": "states-values",
            "values": [{
                "variables": [{"var": "height", "value": 25.0}]
            }]
        }
        
        goal_data = {
            "goal": {
                "exp": {"left": "height", "op": "≤", "right": 100},
                "op": "state-condition"
            },
            "op": "objective"
        }
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        goal_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        failure_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        
        json.dump(jani_data, jani_file, indent=2)
        json.dump(start_data, start_file, indent=2)
        json.dump(goal_data, goal_file, indent=2)
        json.dump(failure_json, failure_file, indent=2)
        
        jani_file.close()
        start_file.close()
        goal_file.close()
        failure_file.close()
        
        try:
            jani = JANI(jani_file.name, start_file.name, goal_file.name, failure_file.name)
            
            # Create test states manually
            var_height_positive = Variable("height", 0, "real", 5.0, "bounded", 100.0, -10.0)
            var_height_zero = Variable("height", 0, "real", 0.0, "bounded", 100.0, -10.0)
            var_height_negative = Variable("height", 0, "real", -2.0, "bounded", 100.0, -10.0)
            
            state_positive = State({"height": var_height_positive})
            state_zero = State({"height": var_height_zero})
            state_negative = State({"height": var_height_negative})
            
            # Test failure evaluation
            assert not jani.failure_reached(state_positive)  # height > 0, not failure
            assert jani.failure_reached(state_zero)          # height = 0, failure
            assert jani.failure_reached(state_negative)      # height < 0, failure
            
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file.name).unlink()
            Path(failure_file.name).unlink()

    def test_complex_conjunction_failure_condition(self):
        """Test complex failure conditions with conjunction and arithmetic."""
        # Test conjunction failure condition: (x + y > 100) AND (z < 5)
        failure_json = {
            "exp": {
                "left": {
                    "left": {
                        "left": "x",
                        "op": "+",
                        "right": "y"
                    },
                    "op": "<",
                    "right": 100
                },
                "op": "∧",
                "right": {
                    "left": "z",
                    "op": "≤",
                    "right": 5
                }
            },
            "op": "state-condition"
        }
        
        jani_data = {
            "jani-version": 1,
            "name": "test-complex-failure",
            "type": "lts",
            "actions": [{"name": "test"}],
            "constants": [],
            "variables": [
                {"name": "x", "type": {"kind": "bounded", "base": "int", "lower-bound": 0, "upper-bound": 100}, "initial-value": 10},
                {"name": "y", "type": {"kind": "bounded", "base": "int", "lower-bound": 0, "upper-bound": 100}, "initial-value": 20},
                {"name": "z", "type": {"kind": "bounded", "base": "int", "lower-bound": 0, "upper-bound": 100}, "initial-value": 30}
            ],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {
            "op": "states-values",
            "values": [{
                "variables": [
                    {"var": "x", "value": 10},
                    {"var": "y", "value": 20},
                    {"var": "z", "value": 30}
                ]
            }]
        }
        
        goal_data = {
            "goal": {
                "exp": {"left": "x", "op": "≤", "right": 100},
                "op": "state-condition"
            },
            "op": "objective"
        }
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        goal_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        failure_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        
        json.dump(jani_data, jani_file, indent=2)
        json.dump(start_data, start_file, indent=2)
        json.dump(goal_data, goal_file, indent=2)
        json.dump(failure_json, failure_file, indent=2)
        
        jani_file.close()
        start_file.close()
        goal_file.close()
        failure_file.close()
        
        try:
            jani = JANI(jani_file.name, start_file.name, goal_file.name, failure_file.name)
            
            # Create test states to verify complex condition
            var_x1 = Variable("x", 0, "int", 40, "bounded", 100, 0)
            var_y1 = Variable("y", 1, "int", 50, "bounded", 100, 0)
            var_z1 = Variable("z", 2, "int", 3, "bounded", 100, 0)
            state1 = State({"x": var_x1, "y": var_y1, "z": var_z1})  # x+y=90<100 AND z=3<=5 -> failure
            
            var_x2 = Variable("x", 0, "int", 60, "bounded", 100, 0)
            var_y2 = Variable("y", 1, "int", 50, "bounded", 100, 0)
            var_z2 = Variable("z", 2, "int", 3, "bounded", 100, 0)
            state2 = State({"x": var_x2, "y": var_y2, "z": var_z2})  # x+y=110>=100 -> no failure
            
            var_x3 = Variable("x", 0, "int", 40, "bounded", 100, 0)
            var_y3 = Variable("y", 1, "int", 50, "bounded", 100, 0)
            var_z3 = Variable("z", 2, "int", 10, "bounded", 100, 0)
            state3 = State({"x": var_x3, "y": var_y3, "z": var_z3})  # x+y=90<100 but z=10>5 -> no failure
            
            # Test complex failure evaluation
            assert jani.failure_reached(state1)      # Both conditions true
            assert not jani.failure_reached(state2)  # First condition false
            assert not jani.failure_reached(state3)  # Second condition false
            
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file.name).unlink()
            Path(failure_file.name).unlink()

    def test_with_actual_example_files(self):
        """Test with actual example files objective.jani and safe.jani."""
        jani_file = "../examples/bouncing_ball/bouncing_ball.jani"
        start_file = "../examples/bouncing_ball/start.jani"
        goal_file = "../examples/bouncing_ball/objective.jani"
        failure_file = "../examples/bouncing_ball/safe.jani"
        
        if not all(Path(f).exists() for f in [jani_file, start_file, goal_file, failure_file]):
            pytest.skip("Required example files not found")
        
        # This should successfully load all files
        jani = JANI(jani_file, start_file, goal_file, failure_file)
        
        # Verify expressions were constructed
        assert jani._goal_expr is not None
        assert jani._failure_expr is not None
        
        # Test with actual states
        for _ in range(10):
            state = jani.reset()
            
            # Should be able to evaluate goal and failure conditions
            goal_result = jani.goal_reached(state)
            failure_result = jani.failure_reached(state)
            
            assert isinstance(goal_result, bool)
            assert isinstance(failure_result, bool)
            
            # For bouncing ball, we shouldn't have both goal and failure simultaneously
            # (though this depends on the specific conditions in the files)

    def test_error_handling_invalid_goal_format(self):
        """Test error handling for invalid goal specification format."""
        # Invalid goal format - missing 'goal' field
        invalid_goal = {
            "exp": {"left": "x", "op": "≤", "right": 10},
            "op": "objective"
        }
        
        jani_data = {
            "jani-version": 1,
            "name": "test-invalid-goal",
            "type": "lts",
            "actions": [{"name": "test"}],
            "constants": [],
            "variables": [{
                "name": "x",
                "type": {"kind": "bounded", "base": "int", "lower-bound": 0, "upper-bound": 100},
                "initial-value": 0
            }],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {
            "op": "states-values",
            "values": [{"variables": [{"var": "x", "value": 0}]}]
        }
        
        failure_data = {
            "exp": {"left": "x", "op": "≤", "right": -1},
            "op": "state-condition"
        }
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        goal_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        failure_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        
        json.dump(jani_data, jani_file, indent=2)
        json.dump(start_data, start_file, indent=2)
        json.dump(invalid_goal, goal_file, indent=2)
        json.dump(failure_data, failure_file, indent=2)
        
        jani_file.close()
        start_file.close()
        goal_file.close()
        failure_file.close()
        
        try:
            with pytest.raises(KeyError):
                JANI(jani_file.name, start_file.name, goal_file.name, failure_file.name)
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file.name).unlink()
            Path(failure_file.name).unlink()

    def test_error_handling_invalid_failure_format(self):
        """Test error handling for invalid failure specification format."""
        # Invalid failure format - wrong operation
        invalid_failure = {
            "exp": {"left": "x", "op": "≤", "right": 0},
            "op": "invalid-operation"
        }
        
        jani_data = {
            "jani-version": 1,
            "name": "test-invalid-failure",
            "type": "lts",
            "actions": [{"name": "test"}],
            "constants": [],
            "variables": [{
                "name": "x",
                "type": {"kind": "bounded", "base": "int", "lower-bound": 0, "upper-bound": 100},
                "initial-value": 0
            }],
            "automata": [{
                "name": "test_automaton",
                "initial-locations": ["loc1"],
                "locations": [{"name": "loc1"}],
                "edges": []
            }]
        }
        
        start_data = {
            "op": "states-values",
            "values": [{"variables": [{"var": "x", "value": 0}]}]
        }
        
        goal_data = {
            "goal": {
                "exp": {"left": "x", "op": "≤", "right": 100},
                "op": "state-condition"
            },
            "op": "objective"
        }
        
        jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        goal_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        failure_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
        
        json.dump(jani_data, jani_file, indent=2)
        json.dump(start_data, start_file, indent=2)
        json.dump(goal_data, goal_file, indent=2)
        json.dump(invalid_failure, failure_file, indent=2)
        
        jani_file.close()
        start_file.close()
        goal_file.close()
        failure_file.close()
        
        try:
            with pytest.raises(ValueError, match="Unsupported safe expression operation"):
                JANI(jani_file.name, start_file.name, goal_file.name, failure_file.name)
        finally:
            Path(jani_file.name).unlink()
            Path(start_file.name).unlink()
            Path(goal_file.name).unlink()
            Path(failure_file.name).unlink()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
