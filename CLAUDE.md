# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
```bash
# Run all tests with verbose output
python -m pytest test_jani.py -v

# Run specific test classes
python -m pytest test_jani.py::TestGuardEvaluation -v
python -m pytest test_jani.py::TestStateTransitions -v
python -m pytest test_jani.py::TestExpressionEvaluation -v

# Run specific test methods
python -m pytest test_jani.py::TestGuardEvaluation::test_simple_boolean_guard_true -v
```

### Running the JANI parser
```bash
# Execute the main JANI parser module
python jani.py

# Run with specific JANI file and start state file
python -c "from jani import JANI; model = JANI('examples/simple_test.jani', 'examples/start.jani')"
```

## Architecture

This is a JANI (JSON interchange format for probabilistic models) parser and simulator written in Python. The system implements a state-based model checker that can parse JANI format files and simulate transitions between states.

### Core Components

**jani.py**: Main implementation containing:
- `Expression` hierarchy: Abstract expression evaluator supporting arithmetic (`+`, `-`, `*`), boolean (`∧`, `∨`, `≤`, `=`), variable references, and constants
- `Variable`/`Constant`: State variables with type information (int, real, bool) and bounds
- `State`: Container for variable dictionaries with vector conversion capabilities  
- `Edge`: Represents transitions with guards and probabilistic destinations
- `Automaton`: Contains edges organized by action labels for state transitions
- `JANI`: Main model class that loads from JSON files and supports initial state generation
- `InitGenerator` classes: Support for different initial state generation strategies (currently `FixedGenerator`)

**Key Design Patterns**:
- Expression evaluation uses the Visitor pattern with recursive JSON construction
- State transitions are probabilistic with deep copying to maintain immutability
- Guards and assignments are evaluated against the original state (not intermediate states)
- Type safety is enforced through dataclasses and type hints

### File Structure

- `jani.py` - Core JANI parser and state machine implementation (~350 lines)
- `test_jani.py` - Comprehensive test suite with 28+ test cases (~1400 lines)
- `examples/` - Sample JANI model files:
  - `simple_test.jani` - Test model with increment/toggle/multiply actions
  - `bouncing_ball.jani` - Physics simulation example
  - `start.jani` - Initial state specification (large file)
- `logs/README_TESTS.md` - Detailed test documentation and coverage information

### Expression System

The expression evaluator supports:
- **Arithmetic**: `+` (addition), `-` (subtraction), `*` (multiplication)
- **Boolean**: `∧` (conjunction), `∨` (disjunction), `≤` (less-than-or-equal), `=` (equality)
- **Variables**: References to state variables by name
- **Constants**: Literal integers, floats, and booleans
- **Nested expressions**: Complex expressions with arbitrary nesting depth

### State Management

States maintain immutability during transitions:
- Original state is preserved via deep copying
- All assignments evaluate against the original state values
- Variables have bounded types with lower/upper bounds
- State vectors can be generated for numerical analysis

### Testing Strategy

The test suite covers:
- Expression evaluation correctness (arithmetic, boolean, constants)
- Guard evaluation with various conditions
- State transitions with single/multiple assignments
- Edge cases (zero values, negatives, boundary conditions)
- Integration tests with real JANI files
- Error handling for invalid inputs