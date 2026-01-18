#ifndef JANI_ENGINE_BASE_COMPONENTS_H
#define JANI_ENGINE_BASE_COMPONENTS_H
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <memory>
#include <boost/container_hash/hash.hpp>


struct Condition {
    enum Operator { LESS_THAN, LESS_EQUAL, EQUAL, GREATER_EQUAL, GREATER_THAN };
    Operator op;
    std::string variable_name;
    // Always cast this value to double as it will be used as an input to NNs
    double value;
    Condition() = default;
    Condition(Operator t, const std::string& var_name, double v) : op(t), variable_name(var_name), value(v) {}
    Condition(const Condition& other) : op(other.op), variable_name(other.variable_name), value(other.value) {}
    Condition& operator=(const Condition& other) {
        if (this != &other) {
            op = other.op;
            variable_name = other.variable_name;
            value = other.value;
        }
        return *this;
    }
};


class Action {
    int id;
    std::string label;
public:
    Action(int id, const std::string& label) : id(id), label(label) {}
    int getId() const { return id; }
    std::string getLabel() const { return label; }
};

class Variable {
    int id;
    std::string name;
    bool is_constant;
public:
    Variable(int id, const std::string& name, bool is_constant) : id(id), name(name), is_constant(is_constant) {}
    virtual ~Variable() = default;
    int getId() const { return id; }
    std::string getName() const { return name; }
    bool isConstant() const { return is_constant; }
    virtual std::unique_ptr<Variable> update(const std::variant<int, double, bool>& val) = 0;
    virtual std::unique_ptr<Variable> clone() = 0;
    virtual void setValue(const std::variant<int, double, bool>& val) = 0;
    virtual std::variant<int, double, bool> getValue() const = 0;
    virtual double getLowerBound() const = 0;
    virtual double getUpperBound() const = 0;
};

class RealVariable: public Variable {
    double value;
    double lower_bound;
    double upper_bound;
public:
    RealVariable(int id, const std::string& name, double lower_bound, double upper_bound, double initial_value)
        : Variable(id, name, false), lower_bound(lower_bound), upper_bound(upper_bound), value(initial_value) {}

    std::variant<int, double, bool> getValue() const { return value; }

    void setValue(const std::variant<int, double, bool>& val) override {
        if (std::holds_alternative<double>(val)) {
            double new_val = std::get<double>(val);
            if (new_val >= lower_bound && new_val <= upper_bound) {
                value = new_val;
            } else {
                throw std::runtime_error("Value out of bounds");
            }
        } else {
            throw std::runtime_error("Invalid type for RealVariable");
        }
    }

    double getLowerBound() const { return lower_bound; }

    double getUpperBound() const { return upper_bound; }

    std::unique_ptr<Variable> update(const std::variant<int, double, bool>& val) {
        // Returns a new RealVariable with updated value
        if (std::holds_alternative<double>(val)) {
            double new_val = std::get<double>(val);
            if (new_val < lower_bound || new_val > upper_bound) {
                throw std::runtime_error("Value out of bounds in RealVariable " + getName() + " update");
            }
            return std::make_unique<RealVariable>(getId(), getName(), lower_bound, upper_bound, new_val);
        }
        throw std::runtime_error("Invalid type for RealVariable " + getName() + " update");
    }

    std::unique_ptr<Variable> clone() {
        // Returns a new RealVariable with the same properties
        return std::make_unique<RealVariable>(getId(), getName(), lower_bound, upper_bound, value);
    }
};

class RealConstant: public Variable {
    double value;
public:
    RealConstant(int id, const std::string& name, double value)
        : Variable(id, name, true), value(value) {}

    std::variant<int, double, bool> getValue() const { return value; }

    void setValue(const std::variant<int, double, bool>& val) override {
        throw std::runtime_error("Cannot set value of a constant variable");
    }


    std::unique_ptr<Variable> update(const std::variant<int, double, bool>& val) {
        throw std::runtime_error("Cannot update a constant variable");
    }

    std::unique_ptr<Variable> clone() {
        // Returns a new RealConstant with the same properties
        return std::make_unique<RealConstant>(getId(), getName(), value);
    }

    double getLowerBound() const { return value; }

    double getUpperBound() const { return value; }
};

class BooleanVariable: public Variable {
    bool value;
public:
    BooleanVariable(int id, const std::string& name, bool initial_value)
        : Variable(id, name, false), value(initial_value) {}

    std::variant<int, double, bool> getValue() const { return value; }

    void setValue(const std::variant<int, double, bool>& val) override {
        if (std::holds_alternative<bool>(val)) {
            value = std::get<bool>(val);
        } else {
            throw std::runtime_error("Invalid type for BooleanVariable");
        }
    }

    double getLowerBound() const { return 0.0; }

    double getUpperBound() const { return 1.0; }

    std::unique_ptr<Variable> update(const std::variant<int, double, bool>& val) {
        // Returns a new BooleanVariable with updated value
        if (std::holds_alternative<bool>(val)) {
            return std::make_unique<BooleanVariable>(getId(), getName(), std::get<bool>(val));
        } else if (std::holds_alternative<int>(val)) {
            int int_val = std::get<int>(val);
            if (int_val == 0)
                return std::make_unique<BooleanVariable>(getId(), getName(), false);
            else if (int_val == 1) 
                return std::make_unique<BooleanVariable>(getId(), getName(), true);
        } else if (std::holds_alternative<double>(val)) {
            double double_val = std::get<double>(val);
            if (double_val == 0.0)
                return std::make_unique<BooleanVariable>(getId(), getName(), false);
            else if (double_val == 1.0)
                return std::make_unique<BooleanVariable>(getId(), getName(), true);
        }
        throw std::runtime_error("Invalid type for BooleanVariable update");
    }

    std::unique_ptr<Variable> clone() {
        // Returns a new BooleanVariable with the same properties
        return std::make_unique<BooleanVariable>(getId(), getName(), value);
    }
};

class BooleanConstant: public Variable {
    bool value;
public:
    BooleanConstant(int id, const std::string& name, bool value)
        : Variable(id, name, true), value(value) {}

    std::variant<int, double, bool> getValue() const { return value; }

    void setValue(const std::variant<int, double, bool>& val) override {
        throw std::runtime_error("Cannot set value of a constant variable");
    }

    std::unique_ptr<Variable> update(const std::variant<int, double, bool>& val) {
        throw std::runtime_error("Cannot update a constant variable");
    }

    std::unique_ptr<Variable> clone() {
        // Returns a new BooleanConstant with the same properties
        return std::make_unique<BooleanConstant>(getId(), getName(), value);
    }

    double getLowerBound() const { return value ? 1.0 : 0.0; }

    double getUpperBound() const { return value ? 1.0 : 0.0; }
};

class IntVariable: public Variable {
    int value;
    int lower_bound;
    int upper_bound;
public:
    IntVariable(int id, const std::string& name, int lower_bound, int upper_bound, int initial_value)
        : Variable(id, name, false), lower_bound(lower_bound), upper_bound(upper_bound), value(initial_value) {}

    std::variant<int, double, bool> getValue() const { return value; }

    void setValue(const std::variant<int, double, bool>& val) override { 
        if (std::holds_alternative<int>(val)) {
            int new_val = std::get<int>(val);
            if (new_val >= lower_bound && new_val <= upper_bound) {
                value = new_val;
            } else {
                throw std::runtime_error("Value out of bounds");
            }
        } else {
            throw std::runtime_error("Invalid type for IntVariable");
        }
    }

    double getLowerBound() const { return static_cast<double>(lower_bound); }

    double getUpperBound() const { return static_cast<double>(upper_bound); }

    std::unique_ptr<Variable> update(const std::variant<int, double, bool>& val) {
        // Returns a new IntVariable with updated value
        if (std::holds_alternative<int>(val)) {
            int new_val = std::get<int>(val);
            if (new_val < lower_bound || new_val > upper_bound) {
                throw std::runtime_error("Value " + std::to_string(new_val) + " out of bounds in IntVariable " + getName() + " update");
            }
            return std::make_unique<IntVariable>(getId(), getName(), lower_bound, upper_bound, new_val);
        } else if (std::holds_alternative<double>(val)) {
            double new_val = std::get<double>(val);
            int int_val = static_cast<int>(new_val);
            if (int_val < lower_bound || int_val > upper_bound) {
                throw std::runtime_error("Value " + std::to_string(int_val) + " out of bounds in IntVariable " + getName() + " update");
            }
            return std::make_unique<IntVariable>(getId(), getName(), lower_bound, upper_bound, int_val);
        }
        throw std::runtime_error("Invalid type for IntVariable " + getName() + " update");
    }

    std::unique_ptr<Variable> clone() {
        // Returns a new IntVariable with the same properties
        return std::make_unique<IntVariable>(getId(), getName(), lower_bound, upper_bound, value);
    }
};

class IntConstant: public Variable {
    int value;
public:
    IntConstant(int id, const std::string& name, int value)     
        : Variable(id, name, true), value(value) {}

    std::variant<int, double, bool> getValue() const { return value; }

    void setValue(const std::variant<int, double, bool>& val) override {
        throw std::runtime_error("Cannot set value of a constant variable");
    }

    std::unique_ptr<Variable> update(const std::variant<int, double, bool>& val) {
        throw std::runtime_error("Cannot update a constant variable");
    }
    
    std::unique_ptr<Variable> clone() {
        // Returns a new IntConstant with the same properties
        return std::make_unique<IntConstant>(getId(), getName(), value);
    }

    double getLowerBound() const { return static_cast<double>(value); }
    
    double getUpperBound() const { return static_cast<double>(value); }
};

class State {
    std::unordered_map<std::string, std::unique_ptr<Variable>> state_values;
public:
    State() = default;

    ~State() = default;

    State(const State& other) {
        for (const auto& pair : other.state_values) {
            state_values[pair.first] = std::move(pair.second->clone());
        }
    }

    State& operator=(const State& other) {
        if (this != &other) {
            state_values.clear();
            for (const auto& pair : other.state_values) {
                state_values[pair.first] = std::move(pair.second->clone());
            }
        }
        return *this;
    }

    void setVariable(const std::string& name, std::unique_ptr<Variable> var) {
        state_values[name] = std::move(var);
    }

    const Variable* getSingleVariable(const std::string& name) const {
        auto it = state_values.find(name);
        if (it != state_values.end()) {
            return it->second.get();
        }
        throw std::runtime_error("Variable not found in state: " + name);
    }

    const std::unordered_map<std::string, std::unique_ptr<Variable>>& getAllVariables() const {
        return state_values;
    }

    bool operator==(const State& other) const {
        if (state_values.size() != other.getAllVariables().size()) {
            throw std::runtime_error("States have different number of variables");
        }
        for (const auto& pair : state_values) {
            const std::string& var_name = pair.first;
            const Variable* var1 = pair.second.get();
            const Variable* var2 = other.getSingleVariable(var_name);
            if (var1->getValue() != var2->getValue()) {
                return false;
            }
        }
        return true;
    }

    std::string toString() const {
        std::vector<std::string> var_strings;
        var_strings.resize(state_values.size());
        for (const auto& pair : state_values) {
            int var_id = pair.second->getId();
            std::string var_name = pair.first;
            std::variant<int, double, bool> var_value = pair.second->getValue();
            std::string value_str;
            if (std::holds_alternative<int>(var_value)) {
                value_str = std::to_string(std::get<int>(var_value));
            } else if (std::holds_alternative<double>(var_value)) {
                value_str = std::to_string(std::get<double>(var_value));
            } else if (std::holds_alternative<bool>(var_value)) {
                value_str = std::get<bool>(var_value) ? "true" : "false";
            }
            std::string var_repr = var_name + " = " + value_str;
            var_strings[var_id] = var_repr;
        }
        std::string state_repr = "[";
        for (size_t i = 0; i < var_strings.size(); ++i) {
            state_repr += var_strings[i];
            if (i < var_strings.size() - 1) {
                state_repr += ", ";
            }
        }
        state_repr += "]";
        return state_repr;
    }

    std::vector<double> toRealVector() const {
        std::vector<double> real_values;
        real_values.resize(state_values.size());
        for (const auto& pair : state_values) {
            int var_id = pair.second->getId();
            std::variant<int, double, bool> var_value = pair.second->getValue();
            if (std::holds_alternative<double>(var_value)) {
                real_values[var_id] = std::get<double>(var_value);
            } else if (std::holds_alternative<int>(var_value)) {
                real_values[var_id] = static_cast<double>(std::get<int>(var_value));
            } else if (std::holds_alternative<bool>(var_value)) {
                real_values[var_id] = std::get<bool>(var_value) ? 1.0 : 0.0;
            } else {
                throw std::runtime_error("Variable type cannot be converted to real number");
            }
        }
        return real_values;
    }

    std::tuple<std::vector<int>, std::vector<double>> toTypedVectors() const {
        std::vector<int> int_values;
        std::vector<double> real_values;
        int_values.resize(state_values.size(), -100000); // Use -100000 as sentinel for uninitialized int values
        real_values.resize(state_values.size(), -100000.0);
        for (const auto& pair : state_values) {
            int var_id = pair.second->getId();
            std::variant<int, double, bool> var_value = pair.second->getValue();
            if (std::holds_alternative<int>(var_value)) {
                int_values[var_id] = std::get<int>(var_value);
            } else if (std::holds_alternative<double>(var_value)) {
                real_values[var_id] = std::get<double>(var_value);
            } else if (std::holds_alternative<bool>(var_value)) {
                int_values[var_id] = std::get<bool>(var_value) ? 1 : 0;
            } else {
                throw std::runtime_error("Variable type cannot be converted to typed vectors");
            }
        }
        return std::make_tuple(int_values, real_values);
    }
};


struct StateHasher {
    std::size_t operator()(const State& s) const noexcept {
        boost::hash<std::tuple<std::vector<int>, std::vector<double>>> tuple_hasher;
        return tuple_hasher(s.toTypedVectors());
    }
};
#endif // JANI_ENGINE_BASE_COMPONENTS_H