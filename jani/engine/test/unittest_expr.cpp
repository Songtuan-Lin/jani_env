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

TEST(ExpressionConditionsExtractTest, EqualityTest) {
    nlohmann::json equality_expr = {
        {"left", "x"},
        {"right", 5},
        {"op", "="}
    };
    Expression* expr = Expression::construct(equality_expr);
    auto conditions = expr->extractConditions();
    // Add assertions to check the extracted conditions
    ASSERT_EQ(conditions.size(), 1);
    EXPECT_EQ(conditions[0].op, Condition::EQUAL);
    EXPECT_EQ(conditions[0].variable_name, "x");
    EXPECT_EQ(conditions[0].value, 5.0);
    delete expr;
}

TEST(ExpressionConditionsExtractTest, EqualityTestBoolConst) {
    nlohmann::json equality_expr = {
        {"left", "flag"},
        {"right", true},
        {"op", "="}
    };
    Expression* expr = Expression::construct(equality_expr);
    auto conditions = expr->extractConditions();
    // Add assertions to check the extracted conditions
    ASSERT_EQ(conditions.size(), 1);
    EXPECT_EQ(conditions[0].op, Condition::EQUAL);
    EXPECT_EQ(conditions[0].variable_name, "flag");
    EXPECT_EQ(conditions[0].value, 1.0);
    delete expr;
}

TEST(ExpressionConditionsExtractTest, LessThanTest) {
    nlohmann::json less_than_expr = {
        {"left", "y"},
        {"right", 10},
        {"op", "<"}
    };
    Expression* expr = Expression::construct(less_than_expr);
    auto conditions = expr->extractConditions();
    // Add assertions to check the extracted conditions
    ASSERT_EQ(conditions.size(), 1);
    EXPECT_EQ(conditions[0].op, Condition::LESS_THAN);
    EXPECT_EQ(conditions[0].variable_name, "y");
    EXPECT_EQ(conditions[0].value, 10.0);
    delete expr;
}

TEST(ExpressionConditionsExtractTest, LessThanOrEqualTest) {
    nlohmann::json less_equal_expr = {
        {"left", "z"},
        {"right", 20.5},
        {"op", "≤"}
    };
    Expression* expr = Expression::construct(less_equal_expr);
    auto conditions = expr->extractConditions();
    // Add assertions to check the extracted conditions
    ASSERT_EQ(conditions.size(), 1);
    EXPECT_EQ(conditions[0].op, Condition::LESS_EQUAL);
    EXPECT_EQ(conditions[0].variable_name, "z");
    EXPECT_EQ(conditions[0].value, 20.5);
    delete expr;
}

TEST(ExpressionConditionsExtractTest, GreaterThanTest) {
    nlohmann::json greater_than_expr = {
        {"left", 15},
        {"right", "a"},
        {"op", "<"}
    };
    Expression* expr = Expression::construct(greater_than_expr);
    auto conditions = expr->extractConditions();
    // Add assertions to check the extracted conditions
    ASSERT_EQ(conditions.size(), 1);
    EXPECT_EQ(conditions[0].op, Condition::GREATER_THAN);
    EXPECT_EQ(conditions[0].variable_name, "a");
    EXPECT_EQ(conditions[0].value, 15.0);
    delete expr;
}

TEST(ExpressionConditionsExtractTest, GreaterThanOrEqualTest) {
    nlohmann::json greater_equal_expr = {
        {"left", 7.5},
        {"right", "b"},
        {"op", "≤"}
    };
    Expression* expr = Expression::construct(greater_equal_expr);
    auto conditions = expr->extractConditions();
    // Add assertions to check the extracted conditions
    ASSERT_EQ(conditions.size(), 1);
    EXPECT_EQ(conditions[0].op, Condition::GREATER_EQUAL);
    EXPECT_EQ(conditions[0].variable_name, "b");
    EXPECT_EQ(conditions[0].value, 7.5);
    delete expr;
}

TEST(ExpressionConditionsExtractTest, AndExpressionTest) {
    nlohmann::json and_expr = {
        {
            "left", {
                {"left", "x"},
                {"right", 5},
                {"op", "="}
            }
        },
        {
            "right", {
                {"left", "y"},
                {"right", 10},
                {"op", "<"}
            }
        },
        {"op", "∧"}
    };
    Expression* expr = Expression::construct(and_expr);
    auto conditions = expr->extractConditions();
    // Add assertions to check the extracted conditions
    ASSERT_EQ(conditions.size(), 2);
    EXPECT_EQ(conditions[0].op, Condition::EQUAL);
    EXPECT_EQ(conditions[0].variable_name, "x");
    EXPECT_EQ(conditions[0].value, 5.0);
    EXPECT_EQ(conditions[1].op, Condition::LESS_THAN);
    EXPECT_EQ(conditions[1].variable_name, "y");
    EXPECT_EQ(conditions[1].value, 10.0);
    delete expr;
}

TEST(ExpressionConditionsExtractTest, NestedAndExpressionTest) {
    nlohmann::json nested_and_expr = nlohmann::json::parse(R"({
        "left": {
            "left": {
                "left": {
                    "left": {
                        "left": "location_load_0",
                        "op": "=",
                        "right": 0
                    },
                    "op": "∧",
                    "right": {
                        "left": "location_load_1",
                        "op": "=",
                        "right": 0
                    }
                },
                "op": "∧",
                "right": {
                    "left": {
                        "left": "location_load_2",
                        "op": "=",
                        "right": 0
                    },
                    "op": "∧",
                    "right": {
                        "left": {
                            "left": "location_load_3",
                            "op": "=",
                            "right": 0
                        },
                        "op": "∧",
                        "right": {
                            "left": "location_load_4",
                            "op": "=",
                            "right": 0
                        }
                    }
                }
            },
            "op": "∧",
            "right": {
                "left": {
                    "left": {
                        "left": "location_load_5",
                        "op": "=",
                        "right": 0
                    },
                    "op": "∧",
                    "right": {
                        "left": {
                            "left": "location_load_6",
                            "op": "=",
                            "right": 0
                        },
                        "op": "∧",
                        "right": {
                            "left": "location_load_7",
                            "op": "=",
                            "right": 0
                        }
                    }
                },
                "op": "∧",
                "right": {
                    "left": {
                        "left": "location_load_8",
                        "op": "=",
                        "right": 0
                    },
                    "op": "∧",
                    "right": {
                        "left": {
                            "left": "location_load_9",
                            "op": "=",
                            "right": 17
                        },
                        "op": "∧",
                        "right": {
                            "left": "truck_0",
                            "op": "=",
                            "right": 9
                        }
                    }
                }
            }
        },
        "op": "∧",
        "right": {
            "left": 0,
            "op": "≤",
            "right": "aux_vel"
        }
    })");
    Expression* expr = Expression::construct(nested_and_expr);
    auto conditions = expr->extractConditions();
    EXPECT_EQ(conditions.size(), 12);
    // Add assertions to check the extracted conditions
    for (auto cond = conditions.begin(); cond != conditions.end(); ++cond) {
        if (cond->variable_name == "location_load_0") {
            EXPECT_EQ(cond->op, Condition::EQUAL);
            EXPECT_EQ(cond->value, 0.0);
        } else if (cond->variable_name == "location_load_1") {
            EXPECT_EQ(cond->op, Condition::EQUAL);
            EXPECT_EQ(cond->value, 0.0);
        } else if (cond->variable_name == "location_load_2") {
            EXPECT_EQ(cond->op, Condition::EQUAL);
            EXPECT_EQ(cond->value, 0.0);
        } else if (cond->variable_name == "location_load_3") {
            EXPECT_EQ(cond->op, Condition::EQUAL);
            EXPECT_EQ(cond->value, 0.0);
        } else if (cond->variable_name == "location_load_4") {
            EXPECT_EQ(cond->op, Condition::EQUAL);
            EXPECT_EQ(cond->value, 0.0);
        } else if (cond->variable_name == "location_load_5") {
            EXPECT_EQ(cond->op, Condition::EQUAL);
            EXPECT_EQ(cond->value, 0.0);
        } else if (cond->variable_name == "location_load_6") {
            EXPECT_EQ(cond->op, Condition::EQUAL);
            EXPECT_EQ(cond->value, 0.0);
        } else if (cond->variable_name == "location_load_7") {
            EXPECT_EQ(cond->op, Condition::EQUAL);
            EXPECT_EQ(cond->value, 0.0);
        } else if (cond->variable_name == "location_load_8") {
            EXPECT_EQ(cond->op, Condition::EQUAL);
            EXPECT_EQ(cond->value, 0.0);
        } else if (cond->variable_name == "location_load_9") {
            EXPECT_EQ(cond->op, Condition::EQUAL);
            EXPECT_EQ(cond->value, 17.0);
        } else if (cond->variable_name == "truck_0") {
            EXPECT_EQ(cond->op, Condition::EQUAL);
            EXPECT_EQ(cond->value, 9.0);
        } else if (cond->variable_name == "aux_vel") {
            EXPECT_EQ(cond->op, Condition::GREATER_EQUAL);
            EXPECT_EQ(cond->value, 0.0);
        }
    }
    std::vector<std::string> expected_var_names = {
        "location_load_0", "location_load_1", "location_load_2", "location_load_3",
        "location_load_4", "location_load_5", "location_load_6", "location_load_7",
        "location_load_8", "location_load_9", "truck_0", "aux_vel"
    };
    for (const auto& var_name : expected_var_names) {
        auto it = std::find_if(conditions.begin(), conditions.end(),
            [&var_name](const Condition& cond) { return cond.variable_name == var_name; });
        EXPECT_NE(it, conditions.end()) << "Condition for variable " << var_name << " not found.";
    }
    delete expr;
}