#include <unordered_set>
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
    std::unordered_map<State, int, StateHasher> cache;
    bool tarjan_dfs(TarjanNode* node, int index,
                    std::vector<TarjanNode*>& stack,
                    std::unordered_set<State, StateHasher>& on_stack);
public:
    TarjanOracle(JANIEngine* eng) : engine(eng) {}
    int isStateSafe(State* state) {
        // Perform Tarjan's algorithm starting from this state
        std::vector<State> stack;
        // Check if a state is on the stack
        std::unordered_map<State, TarjanNode*, StateHasher> on_stack_map;
        TarjanNode *node = new TarjanNode(*state);
        bool safe = tarjan_dfs(node, 0, stack, on_stack_map);
        int result = safe ? 1 : 0;
        cache[*state] = result;
        return result;
    }
}