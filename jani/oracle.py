from .core import *
from dataclasses import dataclass


class TarjanOracle:
    @dataclass
    class Node:
        state: State
        index: int = -1
        lowlink: int = -1


    def __init__(self, model: JANI) -> None:
        self._model = model
        self._safe_states = set()
        self._unsafe_states = set()

    def is_safe(self, state: State) -> bool:
        stack = []
        on_stack = set()
        # Tarjan's algorithm 
        def tarjan(node: TarjanOracle.Node) -> bool:
            assert node.index == -1 and node.lowlink == -1, "Node has already been visited"
            if self._model.goal_reached(node.state) or node.state in self._safe_states:
                return True
            if self._model.failure_reached(node.state) or node.state in self._unsafe_states:
                return False
            node.index = len(stack)
            node.lowlink = node.index
            stack.append(node)
            on_stack.add(node.state)
            safe_state = False
            # Explore successors
            for action_idx in range(self._model.get_action_count()):
                safe_action = True
                action = self._model.get_action(action_idx)
                successors = self._model.get_successors(node.state, action)
                if len(successors) == 0:
                    continue
                for succ_state in successors:
                    next_node = TarjanOracle.Node(succ_state)
                    if next_node.state in on_stack:
                        # Successor is in stack and hence in the current SCC
                        node.lowlink = min(node.lowlink, next_node.lowlink)
                    else:
                        safe_action = tarjan(next_node)
                    if not safe_action:
                        break
                if safe_action:
                    safe_state = True
                    break
            if safe_state:
                self._safe_states.add(node.state)
            else:
                self._unsafe_states.add(node.state)
            if node.lowlink == node.index:
                while True:
                    w = stack.pop()
                    self._safe_states.add(w.state)
                    on_stack.remove(w.state)
                    if w == node:
                        break
            return safe_state
        return tarjan(TarjanOracle.Node(state))