#include <unordered_map>
#include "base_components.h"
#include "engine.h"


struct TarjanNode {
    State state;
    int index;
    int lowlink;
    TarjanNode(State s) : state(s), index(-1), lowlink(-1) {}
};


class TarjanOracle {
    JANIEngine* engine;
    // Cache the safety results for states
    std::unordered_map<State, std::tuple<bool, int>, StateHasher> cache;
    // The main Tarjan's DFS function
    // Returns a tuple of (is_safe, safe_action_id) 
    // safe_action_id is -1 if no safe actions exist (the state is unsafe), the state is a goal state, or the state is a dead-end
    std::tuple<bool, int> tarjan_dfs(TarjanNode* node, int index,
                    std::vector<State>& stack,
                    std::unordered_map<State, std::unique_ptr<TarjanNode>, StateHasher>& on_stack_map);
public:
    TarjanOracle(JANIEngine* eng) : engine(eng) {}
    
    std::tuple<bool, int> stateSafetyWithAction(const State& state) {
        /*Check whether a state is safe. Return the safety result and a safe action starting from the state*/
        #ifndef NDEBUG
        std::cout << "DEBUG: Checking safety for state: " << state.toString() << std::endl;
        #endif
        // Perform Tarjan's algorithm starting from this state
        std::vector<State> stack;
        // Check if a state is on the stack
        std::unordered_map<State, std::unique_ptr<TarjanNode>, StateHasher> on_stack_map;
        std::unique_ptr<TarjanNode> node = std::make_unique<TarjanNode>(state);
        std::tuple<bool, int> result = tarjan_dfs(node.get(), 0, stack, on_stack_map);
        // int safe = std::get<0>(result) ? 1 : 0;
        cache[state] = result;
        #ifndef NDEBUG
        std::cout << "DEBUG: State is marked " << (std::get<0>(result) ? "safe." : "unsafe.") << std::endl;
        #endif
        return result;
    }

    bool isStateSafe(const State& state) {
        std::tuple<bool, int> result = stateSafetyWithAction(state);
        bool safe = std::get<0>(result);
        return safe;
    }

    bool isStateSafeFromVector(const std::vector<double>& state_vector) {
        // Convert vector to State
        State state;
        state = engine->create_state_from_vector(state_vector);
        // Perform Tarjan's algorithm
        return isStateSafe(state);
    }

    bool isEngineStateSafe() {
        const State& state = engine->get_current_state();
        return isStateSafe(state);
    }

    std::tuple<bool, int> engineStateSafetyWithAction() {
        const State& state = engine->get_current_state();
        return stateSafetyWithAction(state);
    }

    // For testing purposes
    const State getEngineCurrentState() {
        return engine->get_current_state();
    }
};