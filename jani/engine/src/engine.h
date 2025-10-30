#ifndef JANI_ENGINE_H
#define JANI_ENGINE_H
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>
#include <random>
#include <filesystem>
#include "nlohmann/json.hpp"
#include "base_components.h"
#include "expressions.h"


class TransitionEdge {
    std::string label; // action label
    Expression* guard;
    std::vector<std::unordered_map<std::string, Expression*>> destinations; // possible outcomes
    std::vector<double> probabilities; // corresponding probabilities
public:
    TransitionEdge(const std::string label, Expression* guard) : label(label), guard(guard) {}
    ~TransitionEdge() {
        delete guard;
        for (auto& dest : destinations)
            for (auto& pair : dest)
                delete pair.second;
    }
    std::string getLabel() const { return label; }
    Expression* getGuard() const { return guard; }
    void addDestination(const nlohmann::json& json_obj) {
        // json_obj is expected to be an element of the "destinations" array of the jani file
        std::unordered_map<std::string, Expression*> assignments;
        for (auto it = json_obj["assignments"].begin(); it != json_obj["assignments"].end(); ++it) {
            // Each assignment is expected to have a "ref" and a "value"
            // "ref" is the variable name, "value" is the expression to assign
            std::string var_name = (*it)["ref"].get<std::string>();
            Expression* expr = Expression::construct((*it)["value"]);
            assignments[var_name] = expr;
        }
        destinations.push_back(assignments);
        if (json_obj.contains("probability")) {
            probabilities.push_back(json_obj["probability"]["exp"].get<double>());
        } else {
            probabilities.push_back(1.0); // Default probability
        }
    }
    bool isEnabled(const State& ctx_state) const {
        auto guard_val = guard->eval(ctx_state);
        if (std::holds_alternative<bool>(guard_val)) {
            return std::get<bool>(guard_val);
        }
        throw std::runtime_error("Guard expression did not evaluate to a boolean");
    }
    State* apply(State& ctx_state, std::mt19937& rng) const {
        // Select a destination based on probabilities
        std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
        int selected = dist(rng);
        const auto& assignments = destinations[selected];

        // Create a new state and apply the assignments
        State* new_state = new State();
        const std::unordered_map<std::string, Variable*>* all_vars = ctx_state.getAllVariables();
        for (const auto& pair : *all_vars) {
            if (assignments.find(pair.first) == assignments.end()) {
                // No assignment for this variable, clone the existing one
                new_state->setVariable(pair.first, pair.second->clone());
            } else {
                // Assignment exists, return the updated variable
                Expression* expr = assignments.at(pair.first);
                Variable* new_variable = pair.second->update(expr->eval(ctx_state));
                new_state->setVariable(pair.first, new_variable);
            }
        }
        return new_state;
    }
};


class Automaton {
    int id;
    std::string location = "loc_0"; // Shouldn't matter for current benchmarks
    // Action labels to transition edges
    std::unordered_map<std::string, std::vector<const TransitionEdge*>> transitions;
public:
    Automaton(const int id) : id(id) {}

    void addTransition(TransitionEdge* transition) {
        transitions[transition->getLabel()].push_back(transition);
    }
};

class JANIEngine {
    // Actions in the model
    std::vector<std::unique_ptr<Action>> actions;
    // Constants in the model
    std::vector<std::unique_ptr<Variable>> constants;
    // Variables in the model
    std::vector<std::unique_ptr<Variable>> variables;
    // Automata in the model (currently only one is supported)
    std::vector<std::unique_ptr<Automaton>> automata;
    // Expression for the objective
    std::unique_ptr<Expression> goal_expression;

    std::unique_ptr<Automaton> constructAutomaton(const nlohmann::json& json_obj, int automaton_id);
    std::unique_ptr<Variable> constructConstant(const nlohmann::json& json_obj);
    std::unique_ptr<Variable> constructVariable(const nlohmann::json& json_obj);
public:
    JANIEngine(
        const std::filesystem::path& jani_model_path, 
        const std::filesystem::path& jani_property_path,
        const std::filesystem::path& start_states_path,
        const std::filesystem::path& objective_path,
        const std::filesystem::path& failure_property_path
    );
    ~JANIEngine() {}
};
#endif // JANI_ENGINE_H