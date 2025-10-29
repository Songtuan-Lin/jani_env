#ifndef JANI_ENGINE_BASE_COMPONENTS_H
#define JANI_ENGINE_BASE_COMPONENTS_H
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>


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
    int getId() const { return id; }
    std::string getName() const { return name; }
    bool isConstant() const { return is_constant; }
    virtual Variable* update(const std::variant<int, double, bool>& val) const = 0;
    virtual Variable* clone() const = 0;
    virtual std::variant<int, double, bool> getValue() const = 0;
};

class RealVariable: public Variable {
    double value;
    double lower_bound;
    double upper_bound;
public:
    RealVariable(int id, const std::string& name, double lower_bound, double upper_bound, double initial_value)
        : Variable(id, name, false), lower_bound(lower_bound), upper_bound(upper_bound), value(initial_value) {}
    std::variant<int, double, bool> getValue() const { return value; }
    void setValue(double val) { 
        if (val >= lower_bound && val <= upper_bound) {
            value = val;
        }
    }
    double getLowerBound() const { return lower_bound; }
    double getUpperBound() const { return upper_bound; }
    Variable* update(const std::variant<int, double, bool>& val) const {
        // Returns a new RealVariable with updated value
        if (std::holds_alternative<double>(val)) {
            double new_val = std::get<double>(val);
            if (new_val < lower_bound || new_val > upper_bound) {
                throw std::runtime_error("Value out of bounds in RealVariable update");
            }
            return new RealVariable(getId(), getName(), lower_bound, upper_bound, new_val);
        }
        throw std::runtime_error("Invalid type for RealVariable update");
    }
    Variable* clone() const {
        // Returns a new RealVariable with the same properties
        return new RealVariable(getId(), getName(), lower_bound, upper_bound, value);
    }
};

class RealConstant: public Variable {
    double value;
public:
    RealConstant(int id, const std::string& name, double value)
        : Variable(id, name, true), value(value) {}
    std::variant<int, double, bool> getValue() const { return value; }
    Variable* update(const std::variant<int, double, bool>& val) const {
        throw std::runtime_error("Cannot update a constant variable");
    }
    Variable* clone() const {
        // Returns a new RealConstant with the same properties
        return new RealConstant(getId(), getName(), value);
    }
};

class BooleanVariable: public Variable {
    bool value;
public:
    BooleanVariable(int id, const std::string& name, bool initial_value)
        : Variable(id, name, false), value(initial_value) {}
    std::variant<int, double, bool> getValue() const { return value; }
    void setValue(bool val) { value = val; }
    Variable* update(const std::variant<int, double, bool>& val) const {
        // Returns a new BooleanVariable with updated value
        if (std::holds_alternative<bool>(val)) {
            return new BooleanVariable(getId(), getName(), std::get<bool>(val));
        }
        throw std::runtime_error("Invalid type for BooleanVariable update");
    }
    Variable* clone() const {
        // Returns a new BooleanVariable with the same properties
        return new BooleanVariable(getId(), getName(), value);
    }
};

class BooleanConstant: public Variable {
    bool value;
public:
    BooleanConstant(int id, const std::string& name, bool value)
        : Variable(id, name, true), value(value) {}
    std::variant<int, double, bool> getValue() const { return value; }
    Variable* update(const std::variant<int, double, bool>& val) const {
        throw std::runtime_error("Cannot update a constant variable");
    }
    Variable* clone() const {
        // Returns a new BooleanConstant with the same properties
        return new BooleanConstant(getId(), getName(), value);
    }
};

class IntVariable: public Variable {
    int value;
    int lower_bound;
    int upper_bound;
public:
    IntVariable(int id, const std::string& name, int lower_bound, int upper_bound, int initial_value)
        : Variable(id, name, false), lower_bound(lower_bound), upper_bound(upper_bound), value(initial_value) {}
    std::variant<int, double, bool> getValue() const { return value; }
    void setValue(int val) { 
        if (val >= lower_bound && val <= upper_bound) {
            value = val;
        }
    }
    int getLowerBound() const { return lower_bound; }
    int getUpperBound() const { return upper_bound; }
    Variable* update(const std::variant<int, double, bool>& val) const {
        // Returns a new IntVariable with updated value
        if (std::holds_alternative<int>(val)) {
            int new_val = std::get<int>(val);
            if (new_val < lower_bound || new_val > upper_bound) {
                throw std::runtime_error("Value out of bounds in IntVariable update");
            }
            return new IntVariable(getId(), getName(), lower_bound, upper_bound, new_val);
        }
        throw std::runtime_error("Invalid type for IntVariable update");
    }
    Variable* clone() const {
        // Returns a new IntVariable with the same properties
        return new IntVariable(getId(), getName(), lower_bound, upper_bound, value);
    }
};

class IntConstant: public Variable {
    int value;
public:
    IntConstant(int id, const std::string& name, int value)     
        : Variable(id, name, true), value(value) {}
    std::variant<int, double, bool> getValue() const { return value; }
    Variable* update(const std::variant<int, double, bool>& val) const {
        throw std::runtime_error("Cannot update a constant variable");
    }
    Variable* clone() const {
        // Returns a new IntConstant with the same properties
        return new IntConstant(getId(), getName(), value);
    }
};

class State {
    std::unordered_map<std::string, Variable*>* state_values;
public:
    State() : state_values(new std::unordered_map<std::string, Variable*>()) {}
    ~State() {
        for (auto& pair : *state_values) {
            delete pair.second;
        }
    }
    void setVariable(const std::string& name, Variable* var) {
        (*state_values)[name] = var;
    }
    Variable* getSingleVariable(const std::string& name) const {
        auto it = state_values->find(name);
        if (it != state_values->end()) {
            return it->second;
        }
        return nullptr;
    }
    const std::unordered_map<std::string, Variable*>* getAllVariables() const {
        return state_values;
    }
};
#endif // JANI_ENGINE_BASE_COMPONENTS_H