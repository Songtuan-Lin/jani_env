#ifndef JANI_ENGINE_EXPRESSIONS_H
#define JANI_ENGINE_EXPRESSIONS_H
#include <string>
#include <unordered_map>
#include <variant>
#include "nlohmann/json.hpp"
#include "base_components.h"


class Expression {
public:
    virtual ~Expression() {}
    virtual std::string toString() const = 0;
    virtual std::variant<int, double, bool> eval(const State& ctx_state) const = 0;
    // TODO: Change to returning unique_ptr later
    static Expression* construct(const nlohmann::json& json_obj);
    // Functions for debugging
    virtual void debugPrintEval(const State& ctx_state) const = 0;
};

class VariableExpression : public Expression {
    std::string name;
public:
    VariableExpression(const std::string& name) : name(name) {}
    std::string toString() const override {
        return name;
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        // Lookup the variable in the context state
        const Variable* var = ctx_state.getSingleVariable(name);
        if (var == nullptr) {
            throw std::runtime_error("Variable not found in context state: " + name);
        }
        return var->getValue();
    }

    void debugPrintEval(const State& ctx_state) const override {
        auto val = eval(ctx_state);
        std::cout << "DEBUG: Variable " << name << " evaluated to " 
                  << std::visit([](auto&& arg) { return std::to_string(arg); }, val) 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};

class IntConstantExpression : public Expression {
    int value;
public:
    IntConstantExpression(int value) : value(value) {}

    std::string toString() const override {
        return std::to_string(value);
    }

    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        return value;
    }

    void debugPrintEval(const State& ctx_state) const override {
        std::cout << "DEBUG: Int constant " << value << " evaluated to " << value 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};

class FloatConstantExpression : public Expression {
    double value;
public:
    FloatConstantExpression(double value) : value(value) {}
    std::string toString() const override {
        return std::to_string(value);
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        return value;
    }
    void debugPrintEval(const State& ctx_state) const override {
        std::cout << "DEBUG: Float constant " << value << " evaluated to " << value 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};

class BooleanConstantExpression : public Expression {
    bool value;
public:
    BooleanConstantExpression(bool value) : value(value) {}
    std::string toString() const override {
        return value ? "true" : "false";
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        return value;
    }
    void debugPrintEval(const State& ctx_state) const override {
        std::cout << "DEBUG: Boolean constant " << (value ? "true" : "false") << " evaluated to " 
                  << (value ? "true" : "false") << " in state " << ctx_state.toString() << std::endl;
    }
};

class AdditionExpression : public Expression {
    Expression* left;
    Expression* right;
public:
    AdditionExpression(Expression* left, Expression* right) : left(left), right(right) {}
    ~AdditionExpression() {
        delete left;
        delete right;
    }
    std::string toString() const override {
        return "(" + left->toString() + " + " + right->toString() + ")";
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        auto left_val = left->eval(ctx_state);
        auto right_val = right->eval(ctx_state);
        if (std::holds_alternative<int>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<int>(left_val) + std::get<int>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<double>(right_val)) {
            return std::get<double>(left_val) + std::get<double>(right_val);
        } else if (std::holds_alternative<int>(left_val) && std::holds_alternative<double>(right_val)) {
            return static_cast<double>(std::get<int>(left_val)) + std::get<double>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<double>(left_val) + static_cast<double>(std::get<int>(right_val));
        }
        throw std::runtime_error("Type mismatch in addition");
    }
    void debugPrintEval(const State& ctx_state) const override {
        left->debugPrintEval(ctx_state);
        right->debugPrintEval(ctx_state);
        auto val = eval(ctx_state);
        std::cout << "DEBUG: Addition expr " << toString() << " evaluated to " 
                  << std::visit([](auto&& arg) { return std::to_string(arg); }, val) 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};

class SubtractionExpression : public Expression {
    Expression* left;
    Expression* right;
public:
    SubtractionExpression(Expression* left, Expression* right) : left(left), right(right) {}
    ~SubtractionExpression() {
        delete left;
        delete right;
    }
    std::string toString() const override {
        return "(" + left->toString() + " - " + right->toString() + ")";
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        auto left_val = left->eval(ctx_state);
        auto right_val = right->eval(ctx_state);
        if (std::holds_alternative<int>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<int>(left_val) - std::get<int>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<double>(right_val)) {
            return std::get<double>(left_val) - std::get<double>(right_val);
        } else if (std::holds_alternative<int>(left_val) && std::holds_alternative<double>(right_val)) {
            return static_cast<double>(std::get<int>(left_val)) - std::get<double>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<double>(left_val) - static_cast<double>(std::get<int>(right_val));
        }
        throw std::runtime_error("Type mismatch in subtraction");
    }
    void debugPrintEval(const State& ctx_state) const override {
        left->debugPrintEval(ctx_state);
        right->debugPrintEval(ctx_state);
        auto val = eval(ctx_state);
        std::cout << "DEBUG: Subtraction expr " << toString() << " evaluated to " 
                  << std::visit([](auto&& arg) { return std::to_string(arg); }, val) 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};

class MultiplicationExpression : public Expression {
    Expression* left;
    Expression* right;
public:
    MultiplicationExpression(Expression* left, Expression* right) : left(left), right(right) {}
    ~MultiplicationExpression() {
        delete left;
        delete right;
    }
    std::string toString() const override {
        return "(" + left->toString() + " * " + right->toString() + ")";
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        auto left_val = left->eval(ctx_state);
        auto right_val = right->eval(ctx_state);
        if (std::holds_alternative<int>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<int>(left_val) * std::get<int>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<double>(right_val)) {
            return std::get<double>(left_val) * std::get<double>(right_val);
        } else if (std::holds_alternative<int>(left_val) && std::holds_alternative<double>(right_val)) {
            return static_cast<double>(std::get<int>(left_val)) * std::get<double>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<double>(left_val) * static_cast<double>(std::get<int>(right_val));
        }
        throw std::runtime_error("Type mismatch in multiplication");
    }
    void debugPrintEval(const State& ctx_state) const override {
        auto val = eval(ctx_state);
        std::cout << "DEBUG: Multiplication expr " << toString() << " evaluated to " 
                  << std::visit([](auto&& arg) { return std::to_string(arg); }, val) 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};

class DivisionExpression : public Expression {
    Expression* left;
    Expression* right;
public:
    DivisionExpression(Expression* left, Expression* right) : left(left), right(right) {}
    ~DivisionExpression() {
        delete left;
        delete right;
    }
    std::string toString() const override {
        return "(" + left->toString() + " / " + right->toString() + ")";
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        auto left_val = left->eval(ctx_state);
        auto right_val = right->eval(ctx_state);
        if (std::holds_alternative<int>(left_val) && std::holds_alternative<int>(right_val)) {
            if (std::get<int>(right_val) == 0) {
                throw std::runtime_error("Division by zero");
            }
            return std::get<int>(left_val) / std::get<int>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<double>(right_val)) {
            if (std::get<double>(right_val) == 0.0) {
                throw std::runtime_error("Division by zero");
            }
            return std::get<double>(left_val) / std::get<double>(right_val);
        } else if (std::holds_alternative<int>(left_val) && std::holds_alternative<double>(right_val)) {
            if (std::get<double>(right_val) == 0.0) {
                throw std::runtime_error("Division by zero");
            }
            return static_cast<double>(std::get<int>(left_val)) / std::get<double>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<int>(right_val)) {
            if (std::get<int>(right_val) == 0) {
                throw std::runtime_error("Division by zero");
            }
            return std::get<double>(left_val) / static_cast<double>(std::get<int>(right_val));
        }
        throw std::runtime_error("Type mismatch in division");
    }
    void debugPrintEval(const State& ctx_state) const override {
        auto val = eval(ctx_state);
        std::cout << "DEBUG: Division expr " << toString() << " evaluated to " 
                  << std::visit([](auto&& arg) { return std::to_string(arg); }, val) 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};

class LessThanExpression : public Expression {
    Expression* left;
    Expression* right;
public:
    LessThanExpression(Expression* left, Expression* right) : left(left), right(right) {}
    ~LessThanExpression() {
        delete left;
        delete right;
    }
    std::string toString() const override {
        return "(" + left->toString() + " < " + right->toString() + ")";
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        auto left_val = left->eval(ctx_state);
        auto right_val = right->eval(ctx_state);
        if (std::holds_alternative<int>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<int>(left_val) < std::get<int>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<double>(right_val)) {
            return std::get<double>(left_val) < std::get<double>(right_val);
        } else if (std::holds_alternative<int>(left_val) && std::holds_alternative<double>(right_val)) {
            return static_cast<double>(std::get<int>(left_val)) < std::get<double>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<double>(left_val) < static_cast<double>(std::get<int>(right_val));
        }
        throw std::runtime_error("Type mismatch in less than comparison");
    }
    void debugPrintEval(const State& ctx_state) const override {
        auto val = eval(ctx_state);
        std::cout << "DEBUG: Less than expr " << toString() << " evaluated to " 
                  << std::visit([](auto&& arg) { return std::to_string(arg); }, val) 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};

class LessThanOrEqualExpression : public Expression {
    Expression* left;
    Expression* right;
public:
    LessThanOrEqualExpression(Expression* left, Expression* right) : left(left), right(right) {}
    ~LessThanOrEqualExpression() {
        delete left;
        delete right;
    }
    std::string toString() const override {
        return "(" + left->toString() + " ≤ " + right->toString() + ")";
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        auto left_val = left->eval(ctx_state);
        auto right_val = right->eval(ctx_state);
        if (std::holds_alternative<int>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<int>(left_val) <= std::get<int>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<double>(right_val)) {
            return std::get<double>(left_val) <= std::get<double>(right_val);
        } else if (std::holds_alternative<int>(left_val) && std::holds_alternative<double>(right_val)) {
            return static_cast<double>(std::get<int>(left_val)) <= std::get<double>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<double>(left_val) <= static_cast<double>(std::get<int>(right_val));
        }
        throw std::runtime_error("Type mismatch in less than or equal comparison");
    }
    void debugPrintEval(const State& ctx_state) const override {
        auto val = eval(ctx_state);
        std::cout << "DEBUG: Less than or equal expr " << toString() << " evaluated to " 
                  << std::visit([](auto&& arg) { return std::to_string(arg); }, val) 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};

class EqualityExpression : public Expression {
    Expression* left;
    Expression* right;
public:
    EqualityExpression(Expression* left, Expression* right) : left(left), right(right) {}
    ~EqualityExpression() {
        delete left;
        delete right;
    }
    std::string toString() const override {
        return "(" + left->toString() + " = " + right->toString() + ")";
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        auto left_val = left->eval(ctx_state);
        auto right_val = right->eval(ctx_state);
        
        if (left_val.index() != right_val.index()) {
            throw std::runtime_error("Type mismatch in equality comparison");
        }
        if (std::holds_alternative<int>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<int>(left_val) == std::get<int>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<double>(right_val)) {
            return std::get<double>(left_val) == std::get<double>(right_val);
        } else if (std::holds_alternative<bool>(left_val) && std::holds_alternative<bool>(right_val)) {
            return std::get<bool>(left_val) == std::get<bool>(right_val);
        } else if (std::holds_alternative<int>(left_val) && std::holds_alternative<double>(right_val)) {
            return static_cast<double>(std::get<int>(left_val)) == std::get<double>(right_val);
        } else if (std::holds_alternative<double>(left_val) && std::holds_alternative<int>(right_val)) {
            return std::get<double>(left_val) == static_cast<double>(std::get<int>(right_val));
        }
        throw std::runtime_error("Unsupported type in equality comparison");
    }
    void debugPrintEval(const State& ctx_state) const override {
        left->debugPrintEval(ctx_state);
        right->debugPrintEval(ctx_state);
        auto val = eval(ctx_state);
        std::cout << "DEBUG: Equality expr " << toString() << " evaluated to " 
                  << std::visit([](auto&& arg) { return std::to_string(arg); }, val) 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};

class AndExpression : public Expression {
    Expression* left;
    Expression* right;
public:
    AndExpression(Expression* left, Expression* right) : left(left), right(right) {}
    ~AndExpression() {
        delete left;
        delete right;
    }
    std::string toString() const override {
        return "(" + left->toString() + " ∧ " + right->toString() + ")";
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        auto left_val = left->eval(ctx_state);
        auto right_val = right->eval(ctx_state);
        
        if (std::holds_alternative<bool>(left_val) && std::holds_alternative<bool>(right_val)) {
            return std::get<bool>(left_val) && std::get<bool>(right_val);
        }
        throw std::runtime_error("Type mismatch in logical AND");
    }
    void debugPrintEval(const State& ctx_state) const override {
        left->debugPrintEval(ctx_state);
        right->debugPrintEval(ctx_state);
        auto val = eval(ctx_state);
        std::cout << "DEBUG: AND expr " << toString() << " evaluated to " 
                  << std::visit([](auto&& arg) { return std::to_string(arg); }, val) 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};

class OrExpression : public Expression {
    Expression* left;
    Expression* right;
public:
    OrExpression(Expression* left, Expression* right) : left(left), right(right) {}
    ~OrExpression() {
        delete left;
        delete right;
    }
    std::string toString() const override {
        return "(" + left->toString() + " ∨ " + right->toString() + ")";
    }
    std::variant<int, double, bool> eval(const State& ctx_state) const override {
        auto left_val = left->eval(ctx_state);
        auto right_val = right->eval(ctx_state);
        if (std::holds_alternative<bool>(left_val) && std::holds_alternative<bool>(right_val)) {
            return std::get<bool>(left_val) || std::get<bool>(right_val);
        }
        throw std::runtime_error("Type mismatch in logical OR");
    }
    void debugPrintEval(const State& ctx_state) const override {
        left->debugPrintEval(ctx_state);
        right->debugPrintEval(ctx_state);
        auto val = eval(ctx_state);
        std::cout << "DEBUG: OR expr " << toString() << " evaluated to " 
                  << std::visit([](auto&& arg) { return std::to_string(arg); }, val) 
                  << " in state " << ctx_state.toString() << std::endl;
    }
};
#endif // JANI_ENGINE_EXPRESSIONS_H