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
        if (engine->reach_failure(node->state))
            return false; // A failure state is an unsafe state
        node->index = index;
        node->lowlink = index;
        stack.push_back(node);
        on_stack_map[node->state] = node;
        
        // Iterate through all applicable actions
        int num_actions = engine->get_num_actions();
        std::vector<bool> action_mask = engine->get_action_mask(node->state)
        bool is_safe_state = true; // If no action is applicable, the state is safe
        for (int action_id = 0; action_id < num_actions; action_id++) {
            if (!action_mask[action_id])
                continue; // Continue to the next action if it is not applicable
            bool is_safe_action = true;
            std::vector<State> successor_states = engine->get_all_successor_states(node->state, action_id);
            for (State& succ_state : successor_states) {
                if (on_stack_map.find(succ_state) != on_stack_map.end()) {
                    // Successor is on stack, update lowlink
                    TarjanNode* succ_node = on_stack_map[succ_state];
                    node->lowlink = std::min(node->lowlink, succ_node->lowlink);
                } else {
                    // Successor not on stack
                    TarjanNode* succ_node = new TarjanNode(succ_state);
                    bool succ_safe = tarjan_dfs(succ_node, index + 1, stack, on_stack_map);
                    if (!succ_safe) {
                        is_safe_action = false;
                        break; // No need to check other successors for this action
                    }
                }
            }
            if (is_safe_action) {
                is_safe_state = true;
                break; // No need to check other actions if one is safe
            } else {
                is_safe_state = false;
            }
        }
        // If node is a root node, pop the stack and generate an SCC
        if (is_safe_state) {
            if (node->lowlink == node->index) {
                cache[node->state] = true; // Mark the state as safe in the cache
                TarjanNode* w = nullptr;
                do {
                    w = stack.back();
                    stack.pop_back();
                    on_stack_map.erase(w->state);
                } while (w->state != node->state);
            }
            return true;
        } else {
            if (node->lowlink == node->index) {
                // Mark the state as unsafe in the cache
                cache[node->state] = false;
                TarjanNode* w = nullptr;
                do {
                    w = stack.back();
                    stack.pop_back();
                    on_stack_map.erase(w->state);
                } while (w->state != node->state);
            }
            return false;
        }
    }