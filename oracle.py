from jani import *
from dataclasses import dataclass


class TarjanOracle:
    @dataclass
    class Node:
        state: State
        index: int = -1
        lowlink: int = -1

        def __hash__(self) -> int:
            return hash(self.state)

    def __init__(self, model: JANI) -> None:
        self._model = model

    def is_safe(self, state: State) -> bool:
        stack = []
        on_stack = set()

        def tarjan(node: TarjanOracle.Node) -> bool:
            assert node.index == -1 and node.lowlink == -1, "Node has already been visited"
            node.index = len(stack)
            node.lowlink = node.index
            stack.append(node)
            on_stack.add(node)

            for action_idx in range(self._model.get_action_count()):
                safe_action = True
                action = self._model.get_action(action_idx)
                successors = self._model.get_successors(node.state, action)
                if len(successors) == 0:
                    continue
                for succ_state in successors:
                    next_node = TarjanOracle.Node(succ_state)
                    if next_node in on_stack:
                        # Successor is in stack and hence in the current SCC
                        node.lowlink = min(node.lowlink, next_node.index)
                    else:
                        safe_action = tarjan(next_node)
                    if not safe_action:
                        break
                