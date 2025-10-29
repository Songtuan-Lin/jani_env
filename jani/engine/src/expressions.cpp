#include "nlohmann/json.hpp"
#include "expressions.h"


// Implementations of Expression methods would go here
Expression* Expression::construct(const nlohmann::json& json_obj) {
    // Implement construction logic based on JSON input
    if (!json_obj.contains("op")) {
        if (json_obj.is_string()) {
            // It's a variable
            return new VariableExpression(json_obj.get<std::string>());
        } else if (json_obj.is_number_integer()) {
            // It's a constant integer
            return new IntConstantExpression(json_obj.get<int>());
        } else if (json_obj.is_number_float()) {
            // It's a constant float
            return new FloatConstantExpression(json_obj.get<double>());
        } else if (json_obj.is_boolean()) {
            // It's a constant boolean
            return new BooleanConstantExpression(json_obj.get<bool>());
        } else {
            throw std::invalid_argument("JSON object without operator but is neither variable nor constant");
        }
        return nullptr;
    }
    // Get the operator
    std::string op = json_obj["op"].get<std::string>();
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