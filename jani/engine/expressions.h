#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <base_components.h>


class Expression {
public:
    virtual ~Expression() {}
    virtual std::string toString() const = 0;
    static Expression* construct(const nlohmann::json& json_obj) {
        // Implement construction logic based on JSON input
        if (!json_obj.contains("op")) {
            if (json_obj.is_string()) {
                // It's a variable
                return new VariableExpression(json_obj.get<std::string>());
            } else if (json_obj.is_number()) {
                // It's a constant number
                return new ConstantExpression(json_obj.get<double>());
            } else {
                throw std::invalid_argument("JSON object without operator but is neither variable nor constant");
            }
            return nullptr;
        }
        // Get the operator
        std::string op = json_obj["op"];
        if (op == "+") {
            // Construct a binary expression for addition
            if (json_obj.contains("left") && json_obj.contains("right")) {
                Expression* left = construct(json_obj["left"]);
                Expression* right = construct(json_obj["right"]);
                return new AdditionExpression(left, right);
            }
        } else if (op == "-") {
            // Construct a binary expression for subtraction
            if (json_obj.contains("left") && json_obj.contains("right")) {
                Expression* left = construct(json_obj["left"]);
                Expression* right = construct(json_obj["right"]);
                return new SubtractionExpression(left, right);
            }
        } else if (op == "*") {
            // Construct a binary expression for multiplication
            if (json_obj.contains("left") && json_obj.contains("right")) {
                Expression* left = construct(json_obj["left"]);
                Expression* right = construct(json_obj["right"]);
                return new MultiplicationExpression(left, right);
            }
        } else if (op == "/") {
            // Construct a binary expression for division
            if (json_obj.contains("left") && json_obj.contains("right")) {
                Expression* left = construct(json_obj["left"]);
                Expression* right = construct(json_obj["right"]);
                return new DivisionExpression(left, right);
            }
        } else if (op == "<") {
            // Construct a binary expression for less than
            if (json_obj.contains("left") && json_obj.contains("right")) {
                Expression* left = construct(json_obj["left"]);
                Expression* right = construct(json_obj["right"]);
                return new LessThanExpression(left, right);
            }
        } else if (op == "≤") {
            // Construct a binary expression for less than or equal
            if (json_obj.contains("left") && json_obj.contains("right")) {
                Expression* left = construct(json_obj["left"]);
                Expression* right = construct(json_obj["right"]);
                return new LessThanOrEqualExpression(left, right);
            }
        } else if (op == "=") {
            // Construct a binary expression for equality
            if (json_obj.contains("left") && json_obj.contains("right")) {
                Expression* left = construct(json_obj["left"]);
                Expression* right = construct(json_obj["right"]);
                return new EqualityExpression(left, right);
            }
        } else if (op == "∧") {
            // Construct a binary expression for logical AND
            if (json_obj.contains("left") && json_obj.contains("right")) {
                Expression* left = construct(json_obj["left"]);
                Expression* right = construct(json_obj["right"]);
                return new AndExpression(left, right);
            }
        } else if (op == "∨") {
            // Construct a binary expression for logical OR
            if (json_obj.contains("left") && json_obj.contains("right")) {
                Expression* left = construct(json_obj["left"]);
                Expression* right = construct(json_obj["right"]);
                return new OrExpression(left, right);
            }
        }
        return nullptr;
    }
};

class VariableExpression : public Expression {
    std::string name;
public:
    VariableExpression(const std::string& name) : name(name) {}
    std::string toString() const override {
        return name;
    }
};

class ConstantExpression : public Expression {
    double value;
public:
    ConstantExpression(double value) : value(value) {}
    std::string toString() const override {
        return std::to_string(value);
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
};