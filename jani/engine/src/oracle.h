#include <unordered_set>
#include "base_components.h"
#include "engine.h"


struct TarjanNode {
    State *state;
    int index;
    int lowlink;
    TarjanNode(State *s) : state(s), index(-1), lowlink(-1) {}
};


class TarjanOracle {
    // Cache the safety results for states
    std::unordered_map<State*, int, StateHasher> cache;
}