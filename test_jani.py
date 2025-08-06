import pytest
import json
import tempfile
from pathlib import Path
from jani import (
    JANI, Edge, State, Variable, Expression, VarExpression, ConstantExpression,
    AddExpression, SubExpression, MulExpression, ConjExpression, DisjExpression,
    LeExpression, EqExpression, Assignment
)


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

    def test_jani_model_loading_and_edge_evaluation(self):
        """Test loading JANI model and evaluating edges."""
        jani_file = self.create_sample_jani_file()

        try:
            # Load JANI model
            jani = JANI(jani_file)

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
            # Clean up temporary file
            Path(jani_file).unlink()

    def test_boundary_conditions_with_jani_model(self):
        """Test boundary conditions using JANI model."""
        jani_file = self.create_sample_jani_file()

        try:
            jani = JANI(jani_file)

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
        jani_file = "examples/simple_test.jani"

        # Check if file exists, skip test if not
        if not Path(jani_file).exists():
            pytest.skip(f"Test JANI file {jani_file} not found")

        jani = JANI(jani_file)

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

    def test_complex_guard_evaluation_with_constants(self):
        """Test guard evaluation involving constants."""
        jani_file = "examples/simple_test.jani"

        if not Path(jani_file).exists():
            pytest.skip(f"Test JANI file {jani_file} not found")

        jani = JANI(jani_file)

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


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
