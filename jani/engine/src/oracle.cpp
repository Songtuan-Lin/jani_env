#include <unordered_map>
#include <fstream>
#include "oracle.h"


size_t rss_mb() {
    std::ifstream f("/proc/self/statm");
    size_t size, resident;
    f >> size >> resident;
    return (resident * 4) / 1024; // MB (page size = 4KB usually)
}


std::tuple<bool, int>TarjanOracle::tarjan_dfs(
    TarjanNode *node, int index, 
    std::vector<State> &stack, 
    std::unordered_map<State, std::unique_ptr<TarjanNode>, StateHasher> &on_stack_map) {
        #ifndef NDEBUG
        std::cout << "DEBUG: Visiting state: " << node->state.toString() << std::endl;
        #endif
        if ((node->index != -1) || (node->lowlink != -1)) 
            throw std::runtime_error("Node should not be initialized before");
        if (cache.find(node->state) != cache.end()) { 
            #ifndef NDEBUG
            std::cout << "  DEBUG: State found in cache." << std::endl;
            std::cout << "  DEBUG: State marked " << (std::get<0>(cache[node->state]) ? "safe." : "unsafe.") << " with action " << std::get<1>(cache[node->state]) << std::endl;
            std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            #endif
            return cache[node->state]; // If the state has been cached
        }
        // TODO: Check whether this inputs the reference of the state
        if (engine->reach_goal(node->state)) {
            #ifndef NDEBUG
            std::cout << "  DEBUG: State is a goal state, marked safe." << std::endl;
            std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            #endif
            return std::make_tuple(true, -1); // A goal state is a safe state
        }
        if (engine->reach_failure(node->state)) {
            #ifndef NDEBUG
            std::cout << "  DEBUG: State is a failure state, marked unsafe." << std::endl;
            std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            #endif
            return std::make_tuple(false, -1); // A failure state is an unsafe state
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
        int safe_action_id = -1; // To record one safe action id
        for (int action_id = 0; action_id < num_actions; action_id++) {
            #ifndef NDEBUG
            std::cout << "  DEBUG: Checking action id " << action_id;
            #endif
            if (!action_mask[action_id]) {
                #ifndef NDEBUG
                std::cout << " -- Action not applicable" << std::endl;
                #endif
                continue; // Continue to the next action if it is not applicable
            }
            #ifndef NDEBUG
            std::cout << " -- Action applicable" << std::endl;
            #endif
            bool is_safe_action = true;
            std::vector<State> successor_states = engine->get_all_successor_states(node->state, action_id);
            #ifndef NDEBUG
            std::cout << "    DEBUG: Number of successor states: " << successor_states.size() << std::endl;
            int succ_idx = 0;
            #endif
            for (State& succ_state : successor_states) {
                if (on_stack_map.find(succ_state) != on_stack_map.end()) {
                    if (cache.find(succ_state) != cache.end()) {
                        bool succ_safe = std::get<0>(cache[succ_state]);
                        #ifndef NDEBUG
                        std::cout << "  DEBUG: Successor state " << succ_idx << " found in cache: " << succ_state.toString() << " marked " << (succ_safe ? "safe." : "unsafe.") << std::endl;
                        succ_idx++;
                        #endif
                        if (!succ_safe) {
                            is_safe_action = false;
                            break; // No need to check other successors for this action
                        } else {
                            continue; // Check the next successor
                        }
                    }
                    #ifndef NDEBUG
                    std::cout << "  DEBUG: Successor state " << succ_idx << " is on stack: " << succ_state.toString() << std::endl;
                    succ_idx++;
                    #endif
                    // Successor is on stack, update lowlink
                    TarjanNode* succ_node = on_stack_map[succ_state].get();
                    node->lowlink = std::min(node->lowlink, succ_node->index);
                    // Update the copy of on-stack map
                    if (on_stack_map.find(node->state) == on_stack_map.end())
                        throw std::runtime_error("Current node not found in on-stack map");
                    on_stack_map[node->state]->lowlink = node->lowlink;
                } else {
                    #ifndef NDEBUG
                    std::cout << "  DEBUG: Successor state " << succ_idx << " is not on stack: " << succ_state.toString() << std::endl;
                    #endif
                    // Successor not on stack
                    std::unique_ptr<TarjanNode> succ_node = std::make_unique<TarjanNode>(succ_state);
                    // TarjanNode* succ_node = new TarjanNode(succ_state);
                    std::tuple<bool, int> succ_result = tarjan_dfs(succ_node.get(), index + 1, stack, on_stack_map);
                    bool succ_safe = std::get<0>(succ_result);
                    #ifndef NDEBUG
                    std::cout << "  DEBUG: Successor state " << succ_idx << " returned " << (succ_safe ? "safe." : "unsafe.") << " with action " << std::get<1>(succ_result) << std::endl;
                    succ_idx++;
                    #endif
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
                safe_action_id = action_id;
                #ifndef NDEBUG
                std::cout << "  DEBUG: State " << node->state.toString() << " with Action id " << action_id << " is safe." << std::endl;
                #endif
                break; // No need to check other actions if one is safe
            } else {
                is_safe_state = false;
            }
        }
        // TODO: We might need remove a SCC after investigating *every action* because SCCs cross different actions should be independent of each other
        // If node is a root node, pop the stack and generate an SCC
        if (is_safe_state) {
            if (node->lowlink == node->index) {
                // std::cout << "            RSS usage before caching: " << rss_mb() << " MB" << std::endl;
                cache[node->state] = std::make_tuple(true, safe_action_id); // Mark the state as safe in the cache
                // std::cout << "            RSS usage after caching: " << rss_mb() << " MB" << std::endl;
                State w;
                do {
                    w = stack.back();
                    stack.pop_back();
                    on_stack_map.erase(w);
                } while (w != node->state);
            }
            #ifndef NDEBUG
            std::cout << "  DEBUG: State marked safe." << std::endl;
            std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            #endif
            return std::make_tuple(true, safe_action_id);
        } else {
            // If state is unsafe, we could always mark it immediately
            cache[node->state] = std::make_tuple(false, -1);
            if (node->lowlink == node->index) {
                // Mark the state as unsafe in the cache
                // std::cout << "            RSS usage before caching: " << rss_mb() << " MB" << std::endl;
                // cache[node->state] = std::make_tuple(false, -1);
                // std::cout << "            RSS usage after caching: " << rss_mb() << " MB" << std::endl;
                State w;
                do {
                    w = stack.back();
                    stack.pop_back();
                    on_stack_map.erase(w);
                } while (w != node->state);
            }
            #ifndef NDEBUG
            std::cout << "  DEBUG: State marked unsafe." << std::endl;
            std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            #endif
            return std::make_tuple(false, -1);
        }
    }