#ifndef JANI_ENGINE_H
#define JANI_ENGINE_H
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>
#include <random>
#include <filesystem>
#include <memory>
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

    State apply(State& ctx_state, std::mt19937& rng) const {
        if (!isEnabled(ctx_state)) {
            throw std::runtime_error("Transition guard is not satisfied in the current state");
        }
        // Select a destination based on probabilities
        std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
        int selected = dist(rng);
        const auto& assignments = destinations[selected];

        // Create a new state and apply the assignments
        State new_state;
        const std::unordered_map<std::string, std::unique_ptr<Variable>> all_vars = ctx_state.getAllVariables();
        for (const auto& pair : all_vars) {
            if (assignments.find(pair.first) == assignments.end()) {
                // No assignment for this variable, clone the existing one
                new_state.setVariable(pair.first, pair.second->clone());
            } else {
                // Assignment exists, return the updated variable
                Expression* expr = assignments.at(pair.first);
                std::unique_ptr<Variable> new_variable = pair.second->update(expr->eval(ctx_state));
                // Be careful to move the unique_ptr
                new_state.setVariable(pair.first, std::move(new_variable));
            }
        }
        return new_state;
    }

    std::vector<State> getAllPossibleOutcomes(State& ctx_state) const {
        if (!isEnabled(ctx_state)) {
            throw std::runtime_error("Transition guard is not satisfied in the current state");
        }
        std::vector<State> outcomes;
        for (const auto& assignments : destinations) {
            State new_state;
            const std::unordered_map<std::string, std::unique_ptr<Variable>> all_vars = ctx_state.getAllVariables();
            for (const auto& pair : all_vars) {
                if (assignments.find(pair.first) == assignments.end()) {
                    // No assignment for this variable, clone the existing one
                    new_state.setVariable(pair.first, pair.second->clone());
                } else {
                    // Assignment exists, return the updated variable
                    Expression* expr = assignments.at(pair.first);
                    std::unique_ptr<Variable> new_variable = pair.second->update(expr->eval(ctx_state));
                    // Be careful to move the unique_ptr
                    new_state.setVariable(pair.first, std::move(new_variable));
                }
            }
            outcomes.push_back(new_state);
        }
        return outcomes;
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

    const std::vector<const TransitionEdge*>* getTransitionsForAction(const std::string& action_label) const {
        auto it = transitions.find(action_label);
        if (it != transitions.end()) {
            return &(it->second);
        }
        throw std::runtime_error("No transitions found for action: " + action_label);
    }
};


class InitStateGenerator {
    // Placeholder for initial state generation logic
    public:
    virtual State* generateInitialState(std::mt19937& rng) const = 0;
    virtual void addInitialState(std::unique_ptr<State> state) = 0;
    virtual const std::vector<std::unique_ptr<State>>& getInitialStatePool() const = 0;
};


class InitStatesFromValues : public InitStateGenerator {
    std::vector<std::unique_ptr<State>> initial_states_pool;
public:
    InitStatesFromValues() = default;

    ~InitStatesFromValues() = default;

    void addInitialState(std::unique_ptr<State> state) override {
        initial_states_pool.push_back(std::move(state));
    }

    State* generateInitialState(std::mt19937& rng) const override {
        if (initial_states_pool.empty()) {
            throw std::runtime_error("No initial states available");
        }
        // Randomly select an initial state from the pool
        std::uniform_int_distribution<> dist(0, initial_states_pool.size() - 1);
        return initial_states_pool[dist(rng)].get();
    }

    const std::vector<std::unique_ptr<State>>& getInitialStatePool() const override {
        return initial_states_pool;
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
    // Expression for the failure property
    std::unique_ptr<Expression> failure_expression;
    // Initial state generator
    std::unique_ptr<InitStateGenerator> init_state_generator;

    // Construct an action
    std::unique_ptr<Action> constructAction(std::string action_label, int index);
    // Construct a single automaton
    std::unique_ptr<Automaton> constructAutomaton(const nlohmann::json& json_obj, int automaton_id);
    // Construct a constant variable
    std::unique_ptr<Variable> constructConstant(const nlohmann::json& json_obj);
    // Construct a variable
    std::unique_ptr<Variable> constructVariable(const nlohmann::json& json_obj);
    // Construct initial state generator from values
    std::unique_ptr<InitStateGenerator> constructGeneratorFromValues(const nlohmann::json& states_array);
    // Construct objective expression
    std::unique_ptr<Expression> constructObjectiveExpression(const nlohmann::json& json_obj);
    // Construct failure expression
    std::unique_ptr<Expression> constructFailureExpression(const nlohmann::json& json_obj);
public:
    JANIEngine() = default;
    JANIEngine(
        const std::filesystem::path& jani_model_path, 
        const std::filesystem::path& jani_property_path,
        const std::filesystem::path& start_states_path,
        const std::filesystem::path& objective_path,
        const std::filesystem::path& failure_property_path
    );
    ~JANIEngine() {}

    // Get the number of actions
    int get_num_actions() {
        return actions.size();
    }

    // Check whether the state reaches the goal
    bool reach_goal(State &s) {
        auto result = goal_expression->eval(s);
        if (!std::holds_alternative<bool>(result))
            throw std::runtime_error("Goal expression does not evaluate to boolean");
        return std::get<bool>(result);
    }

    // Check whether the state is a failure state
    bool reach_failure(State &s) {
        auto result = failure_expression->eval(s);
        if (!std::holds_alternative<bool>(result))
            throw std::runtime_error("Failure expression does not evaluate to boolean");
        return std::get<bool>(result);
    }

    std::vector<bool> get_action_mask(State &s) {
        std::vector<bool> action_mask;
        for (int action_idx = 0; action_idx < actions.size(); action_idx++) {
            std::string action_label = actions[action_idx]->getLabel();
            // TODO: Check the passed argument
            const std::vector<const TransitionEdge*> *transitions = automata[0]->getTransitionsForAction(action_label);
            bool is_enabled = false;
            for (const auto it: *transitions) 
                // If one transition is enabled, the action is applicable
                if (it->isEnabled(s)) {
                    is_enabled = true;
                    break; 
                }
            action_mask.push_back(is_enabled);
        }
        return action_mask;
    }

    std::vector<State> get_all_successor_states(State &s, int action_id) {
        if (action_id < 0 || action_id >= actions.size()) {
            throw std::runtime_error("Invalid action id: " + std::to_string(action_id));
        }
        std::string action_label = actions[action_id]->getLabel();
        const std::vector<const TransitionEdge*> *transitions = automata[0]->getTransitionsForAction(action_label);
        std::vector<State> successor_states;
        int num_transitions = 0;
        for (const auto it: *transitions) {
            if (it->isEnabled(s)) {
                if (num_transitions > 0) {
                    throw std::runtime_error("More than one transition enabled for the same action");
                }
                std::vector<State> outcomes = it->getAllPossibleOutcomes(s);
                successor_states.insert(successor_states.end(), outcomes.begin(), outcomes.end());
                num_transitions++;
            }
        }
        return successor_states;
    }

    // For testing purposes
    std::unique_ptr<InitStateGenerator> testConstructGeneratorFromValues(const nlohmann::json& states_array) {
        return constructGeneratorFromValues(states_array);
    }

    std::unique_ptr<Automaton> testConstructAutomaton(const nlohmann::json& json_obj, int automaton_id) {
        return constructAutomaton(json_obj, automaton_id);
    }

    void testAddAutomaton(std::unique_ptr<Automaton> automaton) {
        automata.push_back(std::move(automaton));
    }

    std::unique_ptr<Variable> testConstructVariable(const nlohmann::json& json_obj) {
        return constructVariable(json_obj);
    }

    void testAddVariable(std::unique_ptr<Variable> variable) {
        variables.push_back(std::move(variable));
    }

    std::unique_ptr<Variable> testConstructConstant(const nlohmann::json& json_obj) {
        return constructConstant(json_obj);
    }
    
    void testAddConstant(std::unique_ptr<Variable> constant) {
        constants.push_back(std::move(constant));
    }

    std::unique_ptr<Action> testConstructAction(std::string action_label, int action_id) {
        return constructAction(action_label, action_id);
    }

    void testAddAction(std::unique_ptr<Action> action) {
        actions.push_back(std::move(action));
    }

    std::unique_ptr<Expression> testConstructObjectiveExpression(const nlohmann::json& json_obj) {
        return constructObjectiveExpression(json_obj);
    }

    void testSetObjectiveExpression(std::unique_ptr<Expression> expr) {
        goal_expression = std::move(expr);
    }

    std::unique_ptr<Expression> testConstructFailureExpression(const nlohmann::json& json_obj) {
        return constructFailureExpression(json_obj);
    }

    void testSetFailureExpression(std::unique_ptr<Expression> expr) {
        failure_expression = std::move(expr);
    }
};
#endif // JANI_ENGINE_H