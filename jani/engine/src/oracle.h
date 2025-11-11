#include <unordered_set>
#include "base_components.h"
#include "engine.h"


struct TarjanNode {
    State *state;
    int index;
    int lowlink;
    bool on_stack;
    TarjanNode(State *s) : state(s), index(-1), lowlink(-1), on_stack(false) {}
};