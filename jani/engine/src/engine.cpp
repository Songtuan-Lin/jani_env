#include <fstream>
#include "nlohmann/json.hpp"
#include "engine.h"


std::unique_ptr<Action> JANIEngine::constructAction(std::string action_label, int action_id) {
    auto action = std::make_unique<Action>(action_id, action_label);
    return action;
}

std::unique_ptr<Automaton> JANIEngine::constructAutomaton(const nlohmann::json& json_obj, int automaton_id) {
    auto automaton = std::make_unique<Automaton>(automaton_id);
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

std::unique_ptr<Variable> JANIEngine::constructConstant(const nlohmann::json& json_obj) {
    // Assume json_obj is an element of the "constants" array
    std::string constant_name = json_obj["name"].get<std::string>();
    std::string constant_type = json_obj["type"].get<std::string>();
    int constant_id = constants.size();
    if (constant_type == "int") {
        int constant_value = json_obj["value"].get<int>();
        return std::make_unique<IntConstant>(constant_id, constant_name, constant_value);
    } else if (constant_type == "real") {
        double constant_value = json_obj["value"].get<double>();
        return std::make_unique<RealConstant>(constant_id, constant_name, constant_value);
    } else if (constant_type == "bool") {
        bool constant_value = json_obj["value"].get<bool>();
        return std::make_unique<BooleanConstant>(constant_id, constant_name, constant_value);
    }
    throw std::runtime_error("Unsupported constant variable type: " + constant_type);
}

std::unique_ptr<Variable> JANIEngine::constructVariable(const nlohmann::json& json_obj) {
    // Assume json_obj is an element of the "variables" array
    std::string variable_name = json_obj["name"].get<std::string>();
    std::string variable_type = json_obj["type"]["base"].get<std::string>();
    int variable_id = constants.size() + variables.size();
    if (variable_type == "int") {
        int lower_bound = json_obj["type"]["lower-bound"].get<int>();
        int upper_bound = json_obj["type"]["upper-bound"].get<int>();
        int initial_value = json_obj["initial-value"].get<int>();
        return std::make_unique<IntVariable>(variable_id, variable_name, lower_bound, upper_bound, initial_value);
    } else if (variable_type == "real") {
        double lower_bound = json_obj["type"]["lower-bound"].get<double>();
        double upper_bound = json_obj["type"]["upper-bound"].get<double>();
        double initial_value = json_obj["initial-value"].get<double>();
        return std::make_unique<RealVariable>(variable_id, variable_name, lower_bound, upper_bound, initial_value);
    } else if (variable_type == "bool") {
        bool initial_value = json_obj["initial-value"].get<bool>();
        return std::make_unique<BooleanVariable>(variable_id, variable_name, initial_value);
    }
    throw std::runtime_error("Unsupported variable type: " + variable_type);
}

std::unique_ptr<InitStateGenerator> JANIEngine::constructGeneratorFromValues(const nlohmann::json& states_array) {

    std::unique_ptr<InitStateGenerator> generator = std::make_unique<InitStatesFromValues>();
    // Iterate through each state
    for (auto it = states_array.begin(); it != states_array.end(); ++it) {
        std::unique_ptr<std::unordered_map<std::string, std::variant<int, double, bool>>> state_values = 
            std::make_unique<std::unordered_map<std::string, std::variant<int, double, bool>>>();
        // Iterate through each variable assignment
        for (auto val_it = (*it)["variables"].begin(); val_it != (*it)["variables"].end(); ++val_it) {
            std::string var_name = (*val_it)["var"].get<std::string>();
            if ((*val_it)["value"].is_number_integer()) {
                int var_value = (*val_it)["value"].get<int>();
                (*state_values)[var_name] = var_value;
            } else if ((*val_it)["value"].is_number_float()) {
                double var_value = (*val_it)["value"].get<double>();
                (*state_values)[var_name] = var_value;
            } else if ((*val_it)["value"].is_boolean()) {
                bool var_value = (*val_it)["value"].get<bool>();
                (*state_values)[var_name] = var_value;
            } else {
                throw std::runtime_error("Unsupported variable value type in start state definition");
            }
        }
        std::unique_ptr<State> state = std::make_unique<State>();
        for (auto const_it = constants.begin(); const_it != constants.end(); ++const_it)
            // Copy constant variables
            state->setVariable((*const_it)->getName(), (*const_it)->clone());
        for (auto var_it = variables.begin(); var_it != variables.end(); ++var_it) {
            std::string var_name = (*var_it)->getName();
            // Check that the variable is defined in the state values
            if (state_values->find(var_name) == state_values->end()) {
                throw std::runtime_error("Variable " + var_name + " missing in start state definition");
            }
            // Copy the variable and set its initial value
            std::unique_ptr<Variable> new_var = (*var_it)->clone();
            switch ((*state_values)[var_name].index()) {
                case 0: // int
                    new_var->setValue(std::get<int>((*state_values)[var_name]));
                    break;
                case 1: // double
                    new_var->setValue(std::get<double>((*state_values)[var_name]));
                    break;
                case 2: // bool
                    new_var->setValue(std::get<bool>((*state_values)[var_name]));
                    break;
                default:
                    throw std::runtime_error("Unsupported variable type in start state assignment");
            }
            state->setVariable(var_name, std::move(new_var));
        }
        // Add the constructed state to the initial states pool
        generator->addInitialState(std::move(state));
    }
    return generator;
}

std::unique_ptr<Expression> JANIEngine::constructObjectiveExpression(const nlohmann::json& json_obj) {
    if (!(json_obj["op"].get<std::string>() == "objective")) 
        throw std::runtime_error("Unsupported objective expression format");
    nlohmann::json goal_section = json_obj["goal"];
    if (!(goal_section["op"].get<std::string>() == "state-condition"))
        throw std::runtime_error("Unsupported objective property format");
    return std::unique_ptr<Expression>(Expression::construct(goal_section["exp"]));
}

std::unique_ptr<Expression> JANIEngine::constructFailureExpression(const nlohmann::json& json_obj) {
    if (!(json_obj["op"].get<std::string>() == "state-condition")) 
        throw std::runtime_error("Unsupported failure expression format");
    nlohmann::json exp_section = json_obj["exp"];
    return std::unique_ptr<Expression>(Expression::construct(exp_section));
}

JANIEngine::JANIEngine(
    const std::filesystem::path& jani_model_path, 
    const std::filesystem::path& jani_property_path,
    const std::filesystem::path& start_states_path,
    const std::filesystem::path& objective_path,
    const std::filesystem::path& failure_property_path,
    int seed
) {
    std::cout << "DEBUG: Initializing JANIEngine with model: " << jani_model_path 
              << ", property: " << jani_property_path 
              << ", start states: " << start_states_path 
              << ", objective: " << objective_path 
              << ", failure property: " << failure_property_path 
              << ", seed: " << seed << std::endl;
    rng = std::mt19937(seed); // Set the random seed
    // Placeholders for file paths
    std::filesystem::path start_file_path, objective_file_path, failure_file_path;
    start_file_path = start_states_path;
    objective_file_path = objective_path;
    failure_file_path = failure_property_path;
    // Load and parse the JANI model file
    std::ifstream model_file(jani_model_path);
    if (!model_file.is_open()) {
        throw std::runtime_error("Failed to open JANI model file: " + jani_model_path.string());
    }
    nlohmann::json jani_json = nlohmann::json::parse(model_file);
    model_file.close();
    std::cout << "DEBUG: Starting to construct JANI model from file: " << jani_model_path << std::endl;
    // Construct actions
    for (auto it = jani_json["actions"].begin(); it != jani_json["actions"].end(); ++it) {
        std::string action_label = (*it)["name"].get<std::string>();
        auto action = constructAction(action_label, actions.size());
        actions.push_back(std::move(action));
    }
    // Construct constants
    for (auto it = jani_json["constants"].begin(); it != jani_json["constants"].end(); ++it) {
        auto constant = constructConstant(*it);
        constants.push_back(std::move(constant));
    }
    // Construct variables
    for (auto it = jani_json["variables"].begin(); it != jani_json["variables"].end(); ++it) {
        auto variable = constructVariable(*it);
        variables.push_back(std::move(variable));
    }
    // Construct automata
    for (auto it = jani_json["automata"].begin(); it != jani_json["automata"].end(); ++it) {
        auto automaton = constructAutomaton(*it, automata.size());
        automata.push_back(std::move(automaton));
    }
    if (automata.size() > 1) {
        throw std::runtime_error("Currently only single automaton models are supported");
    }
    std::cout << "DEBUG: Finished constructing JANI model from file: " << jani_model_path << std::endl;

    // Load the properties of the model
    if (jani_property_path.empty()) {
        if (start_states_path.empty() || objective_path.empty() || failure_property_path.empty()) {
            throw std::runtime_error("Either a JANI property file or all of start states, objective, and failure property files must be provided");
        } 
    } else {
        std::cout << "DEBUG: Starting to load properties from file: " << jani_property_path << std::endl;
        std::ifstream property_file(jani_property_path);
        if (!property_file.is_open()) {
            throw std::runtime_error("Failed to open JANI property file: " + jani_property_path.string());
        }
        nlohmann::json property_json = nlohmann::json::parse(property_file);
        property_file.close();
        nlohmann::json all_properties_json = property_json["properties"];
        if (all_properties_json.size() > 1) {
            throw std::runtime_error("Currently only single property models are supported");
        }
        nlohmann::json property_obj = all_properties_json[0]["expression"];

        // Construct the start states expression
        std::cout << "DEBUG: Starting to construct start states expression from property" << std::endl;
        nlohmann::json start_property = property_obj["start"];
        if (start_property.contains("file"))
            // Load start states from file
            start_file_path = start_property["file"].get<std::string>();
        if (!start_property.contains("op"))
            throw std::runtime_error("Unsupported start states property format");
        // Currently generating initial states based on constraints is not supported
        std::cout << "DEBUG: Finished constructing start states expression from property" << std::endl;

        // Construct the objective expression
        std::cout << "DEBUG: Starting to construct objective expression from property" << std::endl;
        nlohmann::json objective_property = property_obj["objective"];
        if (objective_property.contains("file"))
            // Load objective from file
            objective_file_path = objective_property["file"].get<std::string>();
        else 
            goal_expression = constructObjectiveExpression(objective_property);
        std::cout << "DEBUG: Finished constructing objective expression from property" << std::endl;

        // Construct the failure property expression
        std::cout << "DEBUG: Starting to construct failure property expression from property" << std::endl;
        nlohmann::json failure_property = property_obj["reach"];
        if (failure_property.contains("file"))
            // Load failure property from file
            failure_file_path = failure_property["file"].get<std::string>();
        else 
            failure_expression = constructFailureExpression(failure_property);
        std::cout << "DEBUG: Finished constructing failure property expression from property" << std::endl;

        std::cout << "DEBUG: Finished loading properties from file: " << jani_property_path << std::endl;
    }

    // Load start states from file if specified
    std::cout << "DEBUG: Processing start states" << std::endl;
    if (!start_file_path.empty()) {
        std::cout << "DEBUG: Starting to load start states from file: " << start_file_path << std::endl;
        std::ifstream start_file(start_file_path);
        if (!start_file.is_open()) {
            throw std::runtime_error("Failed to open start states file: " + start_file_path.string());
        }
        nlohmann::json start_states_json = nlohmann::json::parse(start_file);
        start_file.close();
        if (start_states_json["properties"][0]["expression"]["start"]["op"].get<std::string>() == "states-values") {
            // Load start states from values
            nlohmann::json states_array = start_states_json["properties"][0]["expression"]["start"]["values"];
            init_state_generator = constructGeneratorFromValues(states_array);
        } else {
            throw std::runtime_error("Unsupported start states property format in file");
        }
        std::cout << "DEBUG: Finished loading start states from file: " << start_file_path << std::endl;
    }
    // Load objective from file if specified
    std::cout << "DEBUG: Processing objective" << std::endl;
    if (!objective_file_path.empty()) {
        std::cout << "DEBUG: Starting to load objective from file: " << objective_file_path << std::endl;
        std::ifstream objective_file(objective_file_path);
        if (!objective_file.is_open()) {
            throw std::runtime_error("Failed to open objective file: " + objective_file_path.string());
        }
        nlohmann::json objective_json = nlohmann::json::parse(objective_file);
        objective_file.close();
        goal_expression = constructObjectiveExpression(objective_json);
        std::cout << "DEBUG: Finished loading objective from file: " << objective_file_path << std::endl;
    }
    // Load failure property from file if specified
    std::cout << "DEBUG: Processing failure property" << std::endl;
    if (!failure_file_path.empty()) {
        std::cout << "DEBUG: Starting to load failure property from file: " << failure_file_path << std::endl;
        std::ifstream failure_file(failure_file_path);
        if (!failure_file.is_open()) {
            throw std::runtime_error("Failed to open failure property file: " + failure_file_path.string());
        }
        nlohmann::json failure_json = nlohmann::json::parse(failure_file);
        failure_file.close();
        failure_expression = constructFailureExpression(failure_json);
        std::cout << "DEBUG: Finished loading failure property from file: " << failure_file_path << std::endl;
    }
}