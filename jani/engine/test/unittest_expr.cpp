#include <gtest/gtest.h>
#include <string>
#include <format>
#include "nlohmann/json.hpp"
#include "expressions.h"
#include "base_components.h"


TEST(ExpressionConstructTest, VariableConstantExpr) {
    nlohmann::json var_x = nlohmann::json::parse(R"("x")");
    Expression* expr = Expression::construct(var_x);
    EXPECT_EQ(expr->toString(), "x");

    nlohmann::json var_c = nlohmann::json::parse(R"("3")");
    expr = Expression::construct(var_c);
    EXPECT_EQ(expr->toString(), "3");
    delete expr;
}

TEST(ExpressionConstructTest, BinaryOpExpression) {
    std::vector<std::string> ops = {"+", "-", "*", "/", "<", "≤", "∧", "∨", "="};
    Expression* expr;
    for (auto it = ops.begin(); it != ops.end(); ++it) {
        nlohmann::json binary_op = {
            {"left", "x"},
            {"right", "y"},
            {"op", *it}
        };
        expr = Expression::construct(binary_op);
        EXPECT_EQ(expr->toString(), "(x " + *it + " y)");
    }
    delete expr;
}

TEST(ExpressionConstructTest, NestedBinaryExpr) {
    std::vector<std::string> outer_ops = {"+", "-", "*", "/", "<", "≤", "∧", "∨", "="};
    std::vector<std::string> inner_ops_left = {"+", "-", "*", "/", "<", "≤", "∧", "∨", "="};
    std::vector<std::string> inner_ops_right = {"+", "-", "*", "/", "<", "≤", "∧", "∨", "="};
    Expression* expr;
    for (auto outer_it = outer_ops.begin(); outer_it != outer_ops.end(); ++outer_it)
        for (auto inner_it_left = inner_ops_left.begin(); inner_it_left != inner_ops_left.end(); ++inner_it_left)
            for (auto inner_it_right = inner_ops_right.begin(); inner_it_right != inner_ops_right.end(); ++inner_it_right) {
                nlohmann::json nested_binary = {
                    {
                        "left", {
                            {"left", "x"},
                            {"right", 3},
                            {"op", *inner_it_left}
                        }
                    },
                    {
                        "right", {
                            {"left", "y"},
                            {"right", 7},
                            {"op", *inner_it_right}
                        }
                    },
                    {"op", *outer_it}
                };
                expr = Expression::construct(nested_binary);
                EXPECT_EQ(expr->toString(), std::format("((x {} 3) {} (y {} 7))", *inner_it_left, *outer_it, *inner_it_right));
            }
    delete expr;
}

TEST(ExpressionEvalTest, VariableConstantEval) {
    State ctx_state;
    ctx_state.setVariable("x", std::make_unique<IntVariable>(0, "x", 0, 10, 5));
    nlohmann::json var_x = nlohmann::json::parse(R"("x")");
    Expression* expr = Expression::construct(var_x);
    auto val = expr->eval(ctx_state);
    EXPECT_TRUE(std::holds_alternative<int>(val));
    EXPECT_EQ(std::get<int>(val), 5);
    delete expr;

    nlohmann::json var_c = nlohmann::json::parse(R"(3)");
    expr = Expression::construct(var_c);
    val = expr->eval(ctx_state);
    EXPECT_TRUE(std::holds_alternative<int>(val));
    EXPECT_EQ(std::get<int>(val), 3);
    delete expr;
}

TEST(ExpressionEvalTest, AdditionEval) {
    State ctx_state;
    ctx_state.setVariable("x", std::make_unique<IntVariable>(0, "x", 0, 10, 5));
    ctx_state.setVariable("y", std::make_unique<IntVariable>(1, "y", 0, 10, 7));

    nlohmann::json binary_op = {
        {"left", "x"},
        {"right", "y"},
        {"op", "+"}
    };
    Expression* expr = Expression::construct(binary_op);
    auto val = expr->eval(ctx_state);
    EXPECT_TRUE(std::holds_alternative<int>(val));
    EXPECT_EQ(std::get<int>(val), 12);
    delete expr;
}

TEST(ExpressionEvalTest, NestedExpressionEval) {
    State ctx_state;
    ctx_state.setVariable("x", std::make_unique<IntVariable>(0, "x", 0, 10, 5));
    ctx_state.setVariable("y", std::make_unique<IntVariable>(1, "y", 0, 10, 7));

    nlohmann::json nested_binary = {
        {
            "left", {
                {"left", "x"},
                {"right", 3},
                {"op", "+"}
            }
        },
        {
            "right", {
                {"left", "y"},
                {"right", 7},
                {"op", "*"}
            }
        },
        {"op", "-"}
    };
    Expression* expr = Expression::construct(nested_binary);
    auto val = expr->eval(ctx_state);
    EXPECT_TRUE(std::holds_alternative<int>(val));
    EXPECT_EQ(std::get<int>(val), (5 + 3) - (7 * 7)); // 8 - 49 = -41
    delete expr;
}

TEST(ExpressionEvalTest, LogicalExpressionEval) {
    State ctx_state;
    ctx_state.setVariable("x", std::make_unique<IntVariable>(0, "x", 0, 10, 3)); // true
    ctx_state.setVariable("y", std::make_unique<IntVariable>(1, "y", 0, 10, 6)); // false

    nlohmann::json logical_and = {
        {
            "left", {
                {"left", "x"},
                {"right", 5},
                {"op", "<"}
            }
        },
        {
            "right", {
                {"left", "y"},
                {"right", 6},
                {"op", "="}
            }
        },
        {"op", "∧"}
    };
    Expression* expr = Expression::construct(logical_and);
    auto val = expr->eval(ctx_state);
    EXPECT_TRUE(std::holds_alternative<bool>(val));
    EXPECT_EQ(std::get<bool>(val), true); // true AND true = true
    delete expr;

    nlohmann::json logical_or = {
        {
            "left", {
                {"left", 2},
                {"right", "x"},
                {"op", "<"}
            }
        },
        {
            "right", {
                {"left", "y"},
                {"right", 5},
                {"op", "≤"}
            }
        },
        {"op", "∨"}
    };
    expr = Expression::construct(logical_or);
    val = expr->eval(ctx_state);
    EXPECT_TRUE(std::holds_alternative<bool>(val));
    EXPECT_EQ(std::get<bool>(val), true); // false OR true = true
    delete expr;

    nlohmann::json logical_and_false = {
        {
            "left", {
                {"left", 2},
                {"right", "x"},
                {"op", "<"}
            }
        },
        {
            "right", {
                {"left", "y"},
                {"right", 5},
                {"op", "<"}
            }
        },
        {"op", "∧"}
    };
    expr = Expression::construct(logical_and_false);
    val = expr->eval(ctx_state);
    EXPECT_TRUE(std::holds_alternative<bool>(val));
    EXPECT_EQ(std::get<bool>(val), false); // false AND false = false
    delete expr;
}