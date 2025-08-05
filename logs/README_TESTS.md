# JANI Model Parser Test Suite

This document describes the comprehensive unit test suite for the JANI model parser (`jani.py`). The test suite verifies two main functionalities:

1. **Guard evaluation testing** - Verifies whether edge guard conditions are correctly evaluated
2. **State transition testing** - Verifies whether successor states are correctly computed

## Test Structure

The test suite is organized into several test classes:

### TestExpressionEvaluation
Tests the core expression evaluation functionality:
- Variable expression evaluation
- Arithmetic operations (+, -, *)
- Boolean operations (∧, ≤)
- Expression construction from JSON
- Constant expressions (integers, floats, booleans)

### TestGuardEvaluation
Tests guard evaluation functionality:
- Simple boolean guards (true/false)
- Comparison guards with ≤ operator
- Complex guards with arithmetic expressions
- Conjunction guards with ∧ operator
- Edge cases where guards should evaluate to true/false

### TestStateTransitions
Tests state transition functionality:
- Simple variable assignments
- Multiple variable assignments in single transition
- Complex expression assignments
- Guard-controlled transitions
- Boundary condition handling
- State immutability verification

### TestEdgeCases
Tests edge cases and boundary conditions:
- Zero values in expressions
- Negative values in expressions
- Boolean false in conjunction operations
- State immutability during evaluation
- Assignment using original state values

### TestJANIIntegration
Integration tests using complete JANI model data:
- JANI model loading and edge evaluation
- Boundary conditions with full models
- Complex guard evaluation with constants

### TestErrorHandling
Tests error handling for invalid inputs:
- Unsupported operators
- Missing variables in state

### TestWithSimpleJANIFile
Tests using the provided simple test JANI file:
- Model loading verification
- Complex guard evaluation with constants

## Running the Tests

### Prerequisites
```bash
pip install pytest
```

### Run All Tests
```bash
python -m pytest test_jani.py -v
```

### Run Specific Test Classes
```bash
# Test only guard evaluation
python -m pytest test_jani.py::TestGuardEvaluation -v

# Test only state transitions
python -m pytest test_jani.py::TestStateTransitions -v

# Test only expression evaluation
python -m pytest test_jani.py::TestExpressionEvaluation -v
```

### Run Specific Test Methods
```bash
# Test a specific method
python -m pytest test_jani.py::TestGuardEvaluation::test_simple_boolean_guard_true -v
```

## Test Coverage

The test suite covers:

### Guard Evaluation Testing
- ✅ Simple boolean guards (true/false cases)
- ✅ Comparison guards with ≤ operator
- ✅ Complex expressions involving variables and operators
- ✅ Conjunction operations with ∧
- ✅ Edge cases with boundary values
- ✅ Different variable types (int, real, bool)

### State Transition Testing
- ✅ Simple variable assignments
- ✅ Complex update expressions
- ✅ Multiple variable updates in single transition
- ✅ Guard-controlled transitions
- ✅ Boundary conditions
- ✅ State immutability verification
- ✅ Original state value preservation during assignments

### Expression System Testing
- ✅ Variable expressions
- ✅ Constant expressions (int, float, bool)
- ✅ Arithmetic expressions (+, -, *)
- ✅ Boolean expressions (∧, ≤)
- ✅ Nested complex expressions
- ✅ JSON-to-expression construction

### Integration Testing
- ✅ Full JANI model loading
- ✅ Edge evaluation with real model data
- ✅ Constant handling in expressions
- ✅ Boundary condition testing with models

## Test Data

The test suite includes:

1. **Mock test data** - Created programmatically for unit tests
2. **Sample JANI file** - `examples/simple_test.jani` for integration testing
3. **Temporary JANI files** - Generated during integration tests

## Key Test Scenarios

### Guard Evaluation Scenarios
1. **Simple boolean guard**: `"flag"` → true/false
2. **Comparison guard**: `{"op": "≤", "left": "x", "right": "y"}` → true/false
3. **Complex arithmetic guard**: `{"op": "≤", "left": {"op": "+", "left": "x", "right": "y"}, "right": {"op": "*", "left": "x", "right": "y"}}`
4. **Conjunction guard**: `{"op": "∧", "left": condition1, "right": condition2}`

### State Transition Scenarios
1. **Simple assignment**: `x := x + 1`
2. **Multiple assignments**: `x := x * 2, y := y + x`
3. **Complex expression**: `z := (x * z) + (y - 1)`
4. **Conditional transition**: Only execute if guard is true
5. **Boundary testing**: Test at variable bounds

### Edge Cases
1. **Zero values**: Expressions with zero operands
2. **Negative values**: Expressions with negative numbers
3. **Boolean false**: Conjunction with false values
4. **State immutability**: Original state unchanged after transition
5. **Original value usage**: Assignments use pre-transition values

## Expected Test Results

All 28+ tests should pass, covering:
- Expression evaluation correctness
- Guard evaluation accuracy
- State transition correctness
- Edge case handling
- Error condition handling
- Integration with real JANI files

## Extending the Tests

To add new test cases:

1. Add test methods to existing classes for related functionality
2. Create new test classes for new feature areas
3. Follow the naming convention: `test_<functionality>_<scenario>`
4. Include both positive and negative test cases
5. Add docstrings explaining what each test verifies

## Files in Test Suite

- `test_jani.py` - Main test file with all test classes
- `examples/simple_test.jani` - Sample JANI file for integration testing
- `README_TESTS.md` - This documentation file
