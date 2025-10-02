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
        on_stack = dict()
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
            on_stack[node.state] = node
            safe_state = False
            num_applicable_actions = 0
            # Explore successors
            for action_idx in range(self._model.get_action_count()):
                safe_action = True
                action = self._model.get_action(action_idx)
                successors = self._model.get_successors(node.state, action)
                if len(successors) == 0:
                    continue
                num_applicable_actions += 1
                for succ_state in successors:
                    if succ_state in on_stack:
                        loop_node = on_stack[succ_state]
                        assert loop_node.index != -1, "Node on stack has invalid index"
                        assert loop_node.lowlink != -1, "Node on stack has invalid lowlink"
                        node.lowlink = min(node.lowlink, loop_node.lowlink)
                    else:
                        next_node = TarjanOracle.Node(succ_state)
                        safe_action = tarjan(next_node)
                    if not safe_action:
                        break
                if safe_action:
                    safe_state = True
                    break
            if num_applicable_actions == 0:
                safe_state = True
            # if not safe_state:
            #     self._unsafe_states.add(node.state)
            if safe_state:
                if node.lowlink == node.index:
                    self._safe_states.add(node.state)
                    while True:
                        w = stack.pop()
                        r = on_stack.pop(w.state, None)
                        assert r is not None
                        if w == node:
                            break
            else:
                if node.lowlink == node.index:
                    self._unsafe_states.add(node.state)
                    while True:
                        w = stack.pop()
                        # self._safe_states.add(w.state)
                        r = on_stack.pop(w.state, None)
                        assert r is not None
                        if w == node:
                            break
            return safe_state
        return tarjan(TarjanOracle.Node(state))
    