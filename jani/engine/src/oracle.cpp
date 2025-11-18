#include <unordered_map>
#include "oracle.h"


bool TarjanOracle::tarjan_dfs(
    TarjanNode* node, int index, 
    std::vector<TarjanNode*> &stack, 
    std::unordered_map<State, TarjanNode*, StateHasher> &on_stack_map) {
        if ((node->index != -1) || (node->lowlink != -1)) 
            throw std::runtime_error("Node should not be initialized before");
        if (cache.find(node->state) != cache.end()) 
            return cache[node->state]; // If the state has been cached
        // TODO: Check whether this inputs the reference of the state
        if (engine->reach_goal(node->state))
            return true; // A goal state is a safe state
        
        // Iterate through all applicable actions
        int num_actions = engine->get_num_actions();
        std::vector<bool> action_mask = engine->get_action_mask(node->state)
        bool is_safe_state = true;
        for (int action_id = 0; action_id < num_actions; action_id++) {
            if (!action_mask[action_id])
                continue; // Continue to the next action if it is not applicable
        }
    }