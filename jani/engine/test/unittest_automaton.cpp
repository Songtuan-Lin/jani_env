#include <gtest/gtest.h>
#include "engine.h"
#include "expressions.h"
#include "nlohmann/json.hpp"


class AutomatonTest : public ::testing::Test {
protected:
    std::unique_ptr<Automaton> automaton;

    void SetUp() override {
        nlohmann::json json_obj = nlohmann::json::parse(R"({
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
                            "left": "10",
                            "right": x,
                            "op": "<"
                        },
                        "right": {
                            "left": "7",
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
                            },
                            { "ref": "y",
                              "value":
                                {
                                    "left": "y",
                                    "right": 3,
                                    "op": "*"
                                }
                            }
                        ]
                    }
                ]
            }]
        })");
        automaton = std::make_unique<Automaton>(0);
        for (auto it = json_obj["edges"].begin(); it != json_obj["edges"].end(); ++it) {
            std::string label = (*it)["action"].get<std::string>();
            Expression* guard = Expression::construct((*it)["guard"]["exp"]);
            TransitionEdge* edge = new TransitionEdge(label, guard);
            for (auto dest_it = (*it)["destinations"].begin(); dest_it != (*it)["destinations"].end(); ++dest_it) {
                edge->addDestination(*dest_it);
            }
            automaton->addTransition(edge);
        }
    }
};

TEST_F(AutomatonTest, GetTransitionsForAction) {
    const std::vector<const TransitionEdge*>* transitions = automaton->getTransitionsForAction("a");
    EXPECT_EQ(transitions->size(), 1);
    EXPECT_EQ((*transitions)[0]->getLabel(), "a");
}

TEST_F(AutomatonTest, GuardEvaluation) {
    State ctx_state;
    ctx_state.setVariable("x", std::make_unique<IntVariable>(0, "x", 0, 10, 3)); // x = 3

    const std::vector<const TransitionEdge*>* simple_transitions = automaton->getTransitionsForAction("a");
    const std::vector<const TransitionEdge*>* complex_transitions = automaton->getTransitionsForAction("b");

    bool enabled = (*simple_transitions)[0]->isEnabled(ctx_state);
    EXPECT_TRUE(enabled); // Guard x < 5 should be true

    ctx_state.setVariable("x", std::make_unique<IntVariable>(0, "x", 0, 10, 11)); // x = 11
    ctx_state.setVariable("y", std::make_unique<IntVariable>(0, "y", 0, 10, 8)); // y = 8
    enabled = (*complex_transitions)[0]->isEnabled(ctx_state);
    EXPECT_TRUE(enabled); // Guard y < 5 should be true

    ctx_state.setVariable("x", std::make_unique<IntVariable>(0, "x", 0, 10, 6)); // x = 6
    enabled = (*simple_transitions)[0]->isEnabled(ctx_state);
    EXPECT_FALSE(enabled); // Guard x < 5 should be false

    ctx_state.setVariable("x", std::make_unique<IntVariable>(0, "x", 0, 10, 4)); // x = 4
    ctx_state.setVariable("y", std::make_unique<IntVariable>(0, "y", 0, 10, 5)); // y = 5
    enabled = (*complex_transitions)[0]->isEnabled(ctx_state);
    EXPECT_FALSE(enabled); // Guard y < 5 should be false
}