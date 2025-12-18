#include <unordered_map>
#include <fstream>
#include "oracle.h"


size_t rss_mb() {
    std::ifstream f("/proc/self/statm");
    size_t size, resident;
    f >> size >> resident;
    return (resident * 4) / 1024; // MB (page size = 4KB usually)
}


bool TarjanOracle::tarjan_dfs(
    TarjanNode *node, int index, 
    std::vector<State> &stack, 
    std::unordered_map<State, std::unique_ptr<TarjanNode>, StateHasher> &on_stack_map) {
        // std::cout << "DEBUG: Visiting state: " << node->state.toString() << std::endl;
        if ((node->index != -1) || (node->lowlink != -1)) 
            throw std::runtime_error("Node should not be initialized before");
        if (cache.find(node->state) != cache.end()) { 
            // std::cout << "  DEBUG: State found in cache." << std::endl;
            // std::cout << "  DEBUG: State marked " << (cache[node->state] ? "safe." : "unsafe.") << std::endl;
            // std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            return cache[node->state]; // If the state has been cached
        }
        // TODO: Check whether this inputs the reference of the state
        if (engine->reach_goal(node->state)) {
            // std::cout << "  DEBUG: State is a goal state, marked safe." << std::endl;
            // std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            return true; // A goal state is a safe state
        }
        if (engine->reach_failure(node->state)) {
            // std::cout << "  DEBUG: State is a failure state, marked unsafe." << std::endl;
            // std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            return false; // A failure state is an unsafe state
        }
        node->index = index;
        node->lowlink = index;
        stack.push_back(node->state);
        // Create a copy of the node for storing in the on-stack map
        on_stack_map[node->state] = std::make_unique<TarjanNode>(node->state);
        on_stack_map[node->state]->index = node->index;
        on_stack_map[node->state]->lowlink = node->lowlink;
        
        // Iterate through all applicable actions
        int num_actions = engine->get_num_actions();
        std::vector<bool> action_mask = engine->get_action_mask(node->state);
        bool is_safe_state = true; // If no action is applicable, the state is safe
        for (int action_id = 0; action_id < num_actions; action_id++) {
            // std::cout << "  DEBUG: Checking action id " << action_id;
            if (!action_mask[action_id]) {
                // std::cout << " -- Action not applicable" << std::endl;
                continue; // Continue to the next action if it is not applicable
            }
            // std::cout << " -- Action applicable" << std::endl;
            bool is_safe_action = true;
            std::vector<State> successor_states = engine->get_all_successor_states(node->state, action_id);
            for (State& succ_state : successor_states) {
                if (on_stack_map.find(succ_state) != on_stack_map.end()) {
                    // std::cout << "  DEBUG: Successor state is on stack: " << succ_state.toString() << std::endl;
                    // Successor is on stack, update lowlink
                    TarjanNode* succ_node = on_stack_map[succ_state].get();
                    node->lowlink = std::min(node->lowlink, succ_node->index);
                    // Update the copy of on-stack map
                    if (on_stack_map.find(node->state) == on_stack_map.end())
                        throw std::runtime_error("Current node not found in on-stack map");
                    on_stack_map[node->state]->lowlink = node->lowlink;
                } else {
                    // Successor not on stack
                    std::unique_ptr<TarjanNode> succ_node = std::make_unique<TarjanNode>(succ_state);
                    // TarjanNode* succ_node = new TarjanNode(succ_state);
                    bool succ_safe = tarjan_dfs(succ_node.get(), index + 1, stack, on_stack_map);
                    node->lowlink = std::min(node->lowlink, succ_node->lowlink);
                    // Again, update the copy of on-stack map
                    if (on_stack_map.find(node->state) == on_stack_map.end())
                        throw std::runtime_error("Current node not found in on-stack map");
                    on_stack_map[node->state]->lowlink = node->lowlink;
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
                // std::cout << "            RSS usage before caching: " << rss_mb() << " MB" << std::endl;
                cache[node->state] = true; // Mark the state as safe in the cache
                // std::cout << "            RSS usage after caching: " << rss_mb() << " MB" << std::endl;
                State w;
                do {
                    w = stack.back();
                    stack.pop_back();
                    on_stack_map.erase(w);
                } while (w != node->state);
            }
            // std::cout << "  DEBUG: State marked safe." << std::endl;
            // std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            return true;
        } else {
            if (node->lowlink == node->index) {
                // Mark the state as unsafe in the cache
                // std::cout << "            RSS usage before caching: " << rss_mb() << " MB" << std::endl;
                cache[node->state] = false;
                // std::cout << "            RSS usage after caching: " << rss_mb() << " MB" << std::endl;
                State w;
                do {
                    w = stack.back();
                    stack.pop_back();
                    on_stack_map.erase(w);
                } while (w != node->state);
            }
            // std::cout << "  DEBUG: State marked unsafe." << std::endl;
            // std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            return false;
        }
    }