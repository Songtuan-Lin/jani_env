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
    std::unordered_map<State, bool, StateHasher> cache;
    bool tarjan_dfs(TarjanNode* node, int index,
                    std::vector<State>& stack,
                    std::unordered_map<State, TarjanNode*, StateHasher>& on_stack_map);
public:
    TarjanOracle(JANIEngine* eng) : engine(eng) {}
    bool isStateSafe(const State& state) {
        #ifndef NDEBUG
        std::cout << "DEBUG: Checking safety for state: " << state.toString() << std::endl;
        #endif
        // Perform Tarjan's algorithm starting from this state
        std::vector<State> stack;
        // Check if a state is on the stack
        std::unordered_map<State, TarjanNode*, StateHasher> on_stack_map;
        TarjanNode *node = new TarjanNode(state);
        bool safe = tarjan_dfs(node, 0, stack, on_stack_map);
        // int result = safe ? 1 : 0;
        cache[state] = safe;
        #ifndef NDEBUG
        std::cout << "DEBUG: State is marked " << (safe ? "safe." : "unsafe.") << std::endl;
        #endif
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

    // For testing purposes
    const State getEngineCurrentState() {
        return engine->get_current_state();
    }
};