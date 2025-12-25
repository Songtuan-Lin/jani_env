#ifndef JANI_ENGINE_H
#define JANI_ENGINE_H
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>
#include <random>
#include <filesystem>
#include <memory>
#include <cmath>
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
    const Expression* getGuard() const { return guard; }
    const std::vector<std::unordered_map<std::string, Expression*>>& getDestinations() const {
        return destinations;
    }
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

    State apply(const State& ctx_state, std::mt19937& rng) const {
        if (!isEnabled(ctx_state)) {
            throw std::runtime_error("Transition guard is not satisfied in the current state");
        }
        // Select a destination based on probabilities
        std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
        int selected = dist(rng);
        const auto& assignments = destinations[selected];

        // Create a new state and apply the assignments
        State new_state;
        const std::unordered_map<std::string, std::unique_ptr<Variable>>& all_vars = ctx_state.getAllVariables();
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
            const std::unordered_map<std::string, std::unique_ptr<Variable>>& all_vars = ctx_state.getAllVariables();
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
    virtual const State* generateInitialState(std::mt19937& rng) const = 0;
    virtual const State* getState(int index) const = 0;
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

    const State* generateInitialState(std::mt19937& rng) const override {
        if (initial_states_pool.empty()) {
            throw std::runtime_error("No initial states available");
        }
        // Randomly select an initial state from the pool
        std::uniform_int_distribution<> dist(0, initial_states_pool.size() - 1);
        return initial_states_pool[dist(rng)].get();
    }

    const State* getState(int index) const override {
        if (index < 0 || index >= initial_states_pool.size()) {
            throw std::runtime_error("Initial state index out of bounds");
        }
        return initial_states_pool[index].get();
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
    // Current state
    State current_state;
    // Random number generator
    std::mt19937 rng;

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
        const std::filesystem::path& failure_property_path,
        int seed
    );
    ~JANIEngine() {}

    // Get the number of actions
    int get_num_actions() {
        return actions.size();
    }

    int get_num_variables() {
        return variables.size();
    }

    int get_num_constants() {
        return constants.size();
    }

    // Check whether the state reaches the goal
    bool reach_goal(State &s) {
        auto result = goal_expression->eval(s);
        if (!std::holds_alternative<bool>(result))
            throw std::runtime_error("Goal expression does not evaluate to boolean");
        return std::get<bool>(result);
    }

    bool reach_goal_current() {
        return reach_goal(current_state);
    }

    // Check whether the state is a failure state
    bool reach_failure(State &s) {
        auto result = failure_expression->eval(s);
        if (!std::holds_alternative<bool>(result))
            throw std::runtime_error("Failure expression does not evaluate to boolean");
        return std::get<bool>(result);
    }

    bool reach_failure_current() {
        return reach_failure(current_state);
    }

    std::vector<bool> get_action_mask_for_obs(const std::vector<double>& obs) {
        State s = create_state_from_vector(obs);
        return get_action_mask(s);
    }

    std::vector<bool> get_action_mask(const State &s) {
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

    std::vector<bool> get_current_action_mask() {
        return get_action_mask(current_state);
    }

    State create_state_from_vector(const std::vector<double>& values) {
        if (values.size() != constants.size() + variables.size()) {
            throw std::runtime_error("Input vector size does not match number of variables");
        }
        State new_state;
        // Add constants
        for (size_t i = 0; i < constants.size(); i++) {
            auto actual_val = constants[i]->getValue();
            double constant_val;
            if (std::holds_alternative<int>(actual_val)) {
                constant_val = static_cast<double>(std::get<int>(actual_val));
            } else if (std::holds_alternative<double>(actual_val)) {
                constant_val = std::get<double>(actual_val);
            } else if (std::holds_alternative<bool>(actual_val)) {
                constant_val = std::get<bool>(actual_val) ? 1.0 : 0.0;
            } else {
                throw std::runtime_error("Unsupported constant variable type");
            }
            if (std::abs(values[i] - constant_val) > 1e-6) {
                throw std::runtime_error("Constant variable " + constants[i]->getName() + " value mismatch: expected " + std::to_string(constant_val) + ", got " + std::to_string(values[i]));
            }
            new_state.setVariable(constants[i]->getName(), constants[i]->clone());
        }
        // Add variables with values from the input vector
        for (size_t i = 0; i < variables.size(); i++) {
            const auto& var = variables[i];
            std::unique_ptr<Variable> new_var = var->update(values[i + constants.size()]);
            new_state.setVariable(var->getName(), std::move(new_var));
        }
        return new_state;
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

    std::vector<double> reset() {
        // Generate a new initial state
        const State* init_state = init_state_generator->generateInitialState(rng);
        current_state = *init_state;
        #ifndef NDEBUG
        std::cout << "DEBUG: Reset to initial state: " << current_state.toString() << std::endl;
        #endif
        return current_state.toRealVector();
    }

    std::vector<double> reset_with_index(int index) {
        const State* init_state = init_state_generator->getState(index);
        if (init_state == nullptr) {
            throw std::runtime_error("Initial state index out of bounds: " + std::to_string(index));
        }
        current_state = *init_state;
        #ifndef NDEBUG
        std::cout << "DEBUG: Reset to initial state (index " << index << "): " << current_state.toString() << std::endl;
        #endif
        return current_state.toRealVector();
    }

    std::vector<double> step(int action_id) {
        if (action_id < 0 || action_id >= actions.size()) {
            throw std::runtime_error("Invalid action id: " + std::to_string(action_id));
        }
        #ifndef NDEBUG
        std::cout << "DEBUG: Taking step with action id: " << action_id << std::endl;
        #endif
        std::string action_label = actions[action_id]->getLabel();
        const std::vector<const TransitionEdge*> *transitions = automata[0]->getTransitionsForAction(action_label);
        int num_enabled = 0;
        State new_state;
        for (const auto it: *transitions) {
            if (it->isEnabled(current_state)) {
                #ifndef NDEBUG
                std::cout << "DEBUG: Transition with guard " << it->getGuard()->toString() << " is enabled" << std::endl;
                #endif
                if (num_enabled > 0) {
                    #ifdef NDEBUG
                    throw std::runtime_error("More than one transition enabled for the same action");
                    #endif
                    #ifndef NDEBUG
                    it->getGuard()->debugPrintEval(current_state);
                    #endif
                }
                num_enabled++;
                // Apply the transition
                new_state = it->apply(current_state, rng);
            }
        }
        if (num_enabled == 0) {
            throw std::runtime_error("No enabled transition found for action id: " + std::to_string(action_id));
        }
        // Update the current state
        current_state = new_state;
        #ifndef NDEBUG
        if (num_enabled > 1) {
            throw std::runtime_error("More than one transition enabled for the same action");
        }
        std::cout << "DEBUG: New state after step: " << current_state.toString() << std::endl;
        #endif
        return current_state.toRealVector();
    }

    void set_current_state(const State& s) {
        const std::unordered_map<std::string, std::unique_ptr<Variable>>& state_values = s.getAllVariables();
        for (const auto& c : constants) {
            if (state_values.find(c->getName()) == state_values.end()) 
                throw std::runtime_error("Constant variable " + c->getName() + " not found in the provided state");
            if (c->getId() != state_values.at(c->getName())->getId()) 
                throw std::runtime_error("Constant variable " + c->getName() + " ID mismatch");
            if (c->getValue() != state_values.at(c->getName())->getValue()) 
                throw std::runtime_error("Constant variable " + c->getName() + " value mismatch");
        }
        for (const auto& v : variables) {
            if (state_values.find(v->getName()) == state_values.end()) 
                throw std::runtime_error("Variable " + v->getName() + " not found in the provided state");
            if (v->getId() != state_values.at(v->getName())->getId()) 
                throw std::runtime_error("Variable " + v->getName() + " ID mismatch");
        }
        current_state = s;
    }

    const State& get_current_state() const {
        return current_state;
    }

    std::vector<double> get_lower_bounds() {
        std::vector<double> lower_bounds;
        for (const auto& c : constants) {
            if (!(lower_bounds.size() == c->getId())) {
                throw std::runtime_error("Constant variable IDs are not continuous starting from 0");
            }
            std::variant<int, double, bool> val = c->getValue();
            if (std::holds_alternative<double>(val)) {
                lower_bounds.push_back(std::get<double>(val));
            } else if (std::holds_alternative<int>(val)) {
                lower_bounds.push_back(static_cast<double>(std::get<int>(val)));
            } else if (std::holds_alternative<bool>(val)) {
                lower_bounds.push_back(std::get<bool>(val) ? 1.0 : 0.0);
            } 
            else {
                throw std::runtime_error("Constant variable is not of type double for lower bound retrieval");
            }
        }
        for (const auto& v : variables) {
            if (!(lower_bounds.size() == v->getId())) {
                throw std::runtime_error("Variable IDs are not continuous starting from 0 after constants");
            }
            lower_bounds.push_back(v->getLowerBound());
        }
        return lower_bounds;
    }

    std::vector<double> get_upper_bounds() {
        std::vector<double> upper_bounds;
        for (const auto& c : constants) {
            if (!(upper_bounds.size() == c->getId())) {
                throw std::runtime_error("Constant variable IDs are not continuous starting from 0");
            }
            std::variant<int, double, bool> val = c->getValue();
            if (std::holds_alternative<double>(val)) {
                upper_bounds.push_back(std::get<double>(val));;
            } else if (std::holds_alternative<int>(val)) {
                upper_bounds.push_back(static_cast<double>(std::get<int>(val)));
            } else if (std::holds_alternative<bool>(val)) {
                upper_bounds.push_back(std::get<bool>(val) ? 1.0 : 0.0);
            } 
            else {
                throw std::runtime_error("Constant variable is not of type double for upper bound retrieval");
            }
        }
        for (const auto& v : variables) {
            if (!(upper_bounds.size() == v->getId())) {
                throw std::runtime_error("Variable IDs are not continuous starting from 0 after constants");
            }
            upper_bounds.push_back(v->getUpperBound());
        }
        return upper_bounds;
    }

    int get_init_state_pool_size() const {
        return init_state_generator->getInitialStatePool().size();
    }

    std::vector<double> extract_goal_condition() {
        std::vector<Condition> conditions = goal_expression->extractConditions();
        std::vector<double> condition_values;
        // Create a mapping from variable names to conditions (to make lookup easier)
        std::unordered_map<std::string, Condition> condition_map;
        for (const auto& cond : conditions) {
            condition_map[cond.variable_name] = cond;
        }
        // Push all condition values in the order of variables
        for (const auto& var : variables) {
            if (condition_map.find(var->getName()) == condition_map.end())
                continue; // No condition on this variable
            Condition& cond = condition_map[var->getName()];
            condition_values.push_back(cond.value);
        }
        return condition_values;
    }

    std::vector<double> extract_condition_from_state_vector(const std::vector<double>& state_vector) {
        std::vector<Condition> conditions = goal_expression->extractConditions();
        // Create a mapping from variable names to conditions (to make lookup easier)
        std::unordered_map<std::string, Condition> condition_map;
        for (const auto& cond : conditions) {
            condition_map[cond.variable_name] = cond;
        }
        std::vector<double> condition_values;
        size_t num_visited_var = 0; // Used to track number of variables processed
        for (const auto& var : variables) {
            if (condition_map.find(var->getName()) == condition_map.end())
                continue; // No condition on this variable
            num_visited_var++;
            double upper_bound = var->getUpperBound();
            double lower_bound = var->getLowerBound();
            Condition& cond = condition_map[var->getName()];
            double current_var_value = state_vector[var->getId()];
            switch (cond.op) {
                case Condition::LESS_THAN: {
                    if (current_var_value > upper_bound) {
                        throw std::runtime_error("Variable value " + std::to_string(current_var_value) + " is greater than variable " + var->getName() + " upper bound " + std::to_string(upper_bound));
                    }
                    // Randomly sample a value greater than the current variable's value
                    std::uniform_real_distribution<double> dist_le(
                        std::nextafter(current_var_value, std::numeric_limits<double>::infinity()), 
                        upper_bound
                    );
                    condition_values.push_back(dist_le(rng));
                    break;
                }
                case Condition::LESS_EQUAL: {
                    if (current_var_value > upper_bound) {
                        throw std::runtime_error("Variable value " + std::to_string(current_var_value) + " is greater than variable " + var->getName() + " upper bound " + std::to_string(upper_bound));
                    }
                    // Randomly sample a value greater than or equal to the current variable's value
                    std::uniform_real_distribution<double> dist_leq(current_var_value, upper_bound);
                    condition_values.push_back(dist_leq(rng));
                    break;
                }
                case Condition::EQUAL: {
                    if (current_var_value < lower_bound || current_var_value > upper_bound) {
                        throw std::runtime_error("Variable value " + std::to_string(current_var_value) + " is out of bounds for variable " + var->getName());
                    }
                    // Use the current variable's value
                    condition_values.push_back(current_var_value);
                    break;
                }
                case Condition::GREATER_EQUAL: {
                    if (current_var_value < lower_bound) {
                        throw std::runtime_error("Variable value " + std::to_string(current_var_value) + " is less than variable " + var->getName() + " lower bound " + std::to_string(lower_bound));
                    }
                    // Randomly sample a value less than or equal to the current variable's value
                    std::uniform_real_distribution<double> dist_geq(
                        lower_bound, 
                        std::nextafter(current_var_value, std::numeric_limits<double>::infinity())
                    );
                    condition_values.push_back(dist_geq(rng));
                    break;
                }
                case Condition::GREATER_THAN: {
                    if (current_var_value < lower_bound) {
                        throw std::runtime_error("Variable value " + std::to_string(current_var_value) + " is less than variable " + var->getName() + " lower bound " + std::to_string(lower_bound));
                    }
                    // Randomly sample a value less than the current variable's value
                    std::uniform_real_distribution<double> dist_gt(lower_bound, current_var_value);
                    condition_values.push_back(dist_gt(rng));
                    break;
                }
                default:
                    throw std::runtime_error("Unknown condition operator");
            };
        }
        if (num_visited_var != conditions.size())
            throw std::runtime_error("Not all condition variables were found in the state vector");
        return condition_values;
    }

    // For testing purposes
    std::vector<std::string> testGuardsForAction(int action_id) {
        if (action_id < 0 || action_id >= actions.size()) {
            throw std::runtime_error("Invalid action id: " + std::to_string(action_id));
        }
        std::string action_label = actions[action_id]->getLabel();
        const std::vector<const TransitionEdge*> *transitions = automata[0]->getTransitionsForAction(action_label);
        std::vector<std::string> guard_strs;
        for (const auto it: *transitions) {
            guard_strs.push_back(it->getGuard()->toString());
        }
        return guard_strs;
    }

    std::vector<std::vector<std::unordered_map<std::string, std::string>>> testDestinationsForAction(int action_id) {
        if (action_id < 0 || action_id >= actions.size()) {
            throw std::runtime_error("Invalid action id: " + std::to_string(action_id));
        }
        std::string action_label = actions[action_id]->getLabel();
        const std::vector<const TransitionEdge*> *transitions = automata[0]->getTransitionsForAction(action_label);
        std::vector<std::vector<std::unordered_map<std::string, std::string>>> all_destinations;
        for (const auto it: *transitions) {
            std::vector<std::unordered_map<std::string, std::string>> dests_for_transition;
            const auto& destinations = it->getDestinations();
            for (const auto& assignment_map : destinations) {
                std::unordered_map<std::string, std::string> assignment_str_map;
                for (const auto& pair : assignment_map) {
                    assignment_str_map[pair.first] = pair.second->toString();
                }
                dests_for_transition.push_back(assignment_str_map);
            }
            all_destinations.push_back(dests_for_transition);
        }
        return all_destinations;
    }

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