#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "engine.h"
#include "base_components.h"


class OracleTest : public ::testing::Test {
protected:
    JANIEngine *engine;
    TarjanOracle *oracle;

    void SetUp() override {
        engine = new JANIEngine();
        // Add some variables to the engine
        nlohmann::json variables_json = nlohmann::json::parse(R"([
            {
                "name": "x",
                "type": {
                    "base": "int",
                    "lower-bound": 0,
                    "upper-bound": 10
                },
                "initial-value": 0
            },
            {
                "name": "y",
                "type": {
                    "base": "int",
                    "lower-bound": 0,
                    "upper-bound": 10
                },
                "initial-value": 0
            }
        ])");
        for (auto it = variables_json.begin(); it != variables_json.end(); ++it) {
            std::unique_ptr<Variable> var = engine->testConstructVariable(*it);
            engine->testAddVariable(std::move(var));
        }
        // Add actions to the engine
        std::vector<std::string> action_labels = {"a", "b"};
        for (int i = 0; i < action_labels.size(); i++) {
            std::unique_ptr<Action> action = engine->testConstructAction(action_labels[i], i);
            engine->testAddAction(std::move(action));
        }
        // Add a simple automaton to the engine
        nlohmann::json automaton_json = nlohmann::json::parse(R"({
            "edges": [
            {
                "action": "a",
                "guard": { 
                    "exp": {
                        "left": "x",
                        "right": 5,
                        "op": "<"
                    }
                },
                "destinations": [
                    {
                        "assignments": [
                            { "ref": "x", 
                              "value":  
                                {
                                    "left": "x",
                                    "right": 1,
                                    "op": "+"
                                }
                            }
                        ]
                    }
                ]
            },
            {
                "action": "b",
                "guard": { 
                    "exp": {
                        "left": {
                            "left": 5,
                            "right": "x",
                            "op": "≤"
                        },
                        "right": {
                            "left": 7,
                            "right": "y",
                            "op": "<"
                        },
                        "op": "∧"
                    }
                },
                "destinations": [
                    {
                        "assignments": [
                            { "ref": "x", 
                              "value":  
                                {
                                    "left": "x",
                                    "right": 2,
                                    "op": "-"
                                }
                            }
                        ]
                    },
                    {
                        "assignments": [
                            { "ref": "y",
                              "value":
                                {
                                    "left": "y",
                                    "right": 1,
                                    "op": "+"
                                }
                            }
                        ]
                    }
                ]
            }
            ]
        })");
        std::unique_ptr<Automaton> automaton = engine->testConstructAutomaton(automaton_json, 0);
        engine->testAddAutomaton(std::move(automaton));
        // Set objective and failure expressions
        nlohmann::json objective_json = nlohmann::json::parse(R"({
            "goal": {
                "exp": {
                    "left": "x",
                    "right": 8,
                    "op": "="
                },
                "op": "state-condition"
            },
            "op": "objective"
        })");
        std::unique_ptr<Expression> objective_expr = engine->testConstructObjectiveExpression(objective_json);
        engine->testSetObjectiveExpression(std::move(objective_expr));
        nlohmann::json failure_json = nlohmann::json::parse(R"({
            "exp": {
                "left": 7,
                "right": "y",
                "op": "≤"
            },
            "op": "state-condition"
        })");
        std::unique_ptr<Expression> failure_expr = engine->testConstructFailureExpression(failure_json);
        engine->testSetFailureExpression(std::move(failure_expr));
        // Create the oracle
        oracle = new TarjanOracle(engine);
    }
};

TEST_F(OracleTest, StateSafety) {
    State *s = new State();
    s->setVariable("x", std::make_unique<IntVariable>(0, "x", 0, 10, 3)); // x = 3
    s->setVariable("y", std::make_unique<IntVariable>(1, "y", 0, 10, 6)); // y = 6
    bool is_safe = oracle->isStateSafe(*s);
    EXPECT_FALSE(is_safe); // (x < 5) is true, so action "
}