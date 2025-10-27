#include <vector>
#include <string>
#include <unordered_map>


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
};

class RealVariable: public Variable {
    double value;
    double lower_bound;
    double upper_bound;
public:
    RealVariable(int id, const std::string& name, double lower_bound, double upper_bound, double initial_value)
        : Variable(id, name, false), lower_bound(lower_bound), upper_bound(upper_bound), value(initial_value) {}
    double getValue() const { return value; }
    void setValue(double val) { 
        if (val >= lower_bound && val <= upper_bound) {
            value = val;
        }
    }
    double getLowerBound() const { return lower_bound; }
    double getUpperBound() const { return upper_bound; }
};

class RealConstant: public Variable {
    double value;
public:
    RealConstant(int id, const std::string& name, double value)
        : Variable(id, name, true), value(value) {}
    double getValue() const { return value; }
};

class BooleanVariable: public Variable {
    bool value;
public:
    BooleanVariable(int id, const std::string& name, bool initial_value)
        : Variable(id, name, false), value(initial_value) {}
    bool getValue() const { return value; }
    void setValue(bool val) { value = val; }
};

class BooleanConstant: public Variable {
    bool value;
public:
    BooleanConstant(int id, const std::string& name, bool value)
        : Variable(id, name, true), value(value) {}
    bool getValue() const { return value; }
};

class IntVariable: public Variable {
    int value;
    int lower_bound;
    int upper_bound;
public:
    IntVariable(int id, const std::string& name, int lower_bound, int upper_bound, int initial_value)
        : Variable(id, name, false), lower_bound(lower_bound), upper_bound(upper_bound), value(initial_value) {}
    int getValue() const { return value; }
    void setValue(int val) { 
        if (val >= lower_bound && val <= upper_bound) {
            value = val;
        }
    }
    int getLowerBound() const { return lower_bound; }
    int getUpperBound() const { return upper_bound; }
};

class IntConstant: public Variable {
    int value;
public:
    IntConstant(int id, const std::string& name, int value)     
        : Variable(id, name, true), value(value) {}
    int getValue() const { return value; }
};

class State {
    std::unique_ptr<std::unordered_map<std::string, Variable*>> state_values;
};

class Engine {
    std::vector<Action*> actions;
    std::vector<Variable*> constants;
    std::vector<Variable*> variables;
public:
    ~Engine() {
        for (auto action : actions) {
            delete action;
        }
        for (auto var : variables) {
            delete var;
        }
        for (auto const_var : constants) {
            delete const_var;
        }
    }
};