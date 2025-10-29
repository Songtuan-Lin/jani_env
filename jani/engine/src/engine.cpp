#include "nlohmann/json.hpp"
#include "engine.h"


Automaton* JANIEngine::constructAutomaton(const nlohmann::json& json_obj, int automaton_id) {
    Automaton* automaton = new Automaton(automaton_id);
    // Assume json_obj is an element of "automata" array
    for (auto it = json_obj["edges"].begin(); it != json_obj["edges"].end(); ++it) {
        std::string label = (*it)["action"].get<std::string>();
        Expression* guard = Expression::construct((*it)["guard"]["exp"]);
        TransitionEdge* edge = new TransitionEdge(label, guard);
        // Add all possible outcomes to the edge
        for (auto dest_it = (*it)["destinations"].begin(); dest_it != (*it)["destinations"].end(); ++dest_it) {
            edge->addDestination(*dest_it);
        }
        automaton->addTransition(edge);
    }
    return automaton;
}