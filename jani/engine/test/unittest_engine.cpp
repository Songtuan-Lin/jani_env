#include <gtest/gtest.h>
#include <random>
#include <nlohmann/json.hpp>
#include "engine.h"


class AutomatonTest : public ::testing::Test {
protected:
    std::unique_ptr<InitStateGenerator> init_generator;
    std::mt19937 rng(42); // Fixed seed
    void SetUp() override {
        JANIEngine engine;
        // Add some variables to the engine
        nlohmann::json variables_json = nlohmann::json::parse(R"([
            {
                "name": "episode",
                "type": {
                    "base": "int",
                    "lower-bound": 0,
                    "upper-bound": 4000
                },
                "initial-value": 0
            },
            {
                "name": "height",
                "type": {
                    "base": "real",
                    "lower-bound": 0,
                    "upper-bound": 1000
                },
                "initial-value": 5
            },
            {
                "name": "velocity",
                "type": {
                    "base": "real",
                    "lower-bound": -100,
                    "upper-bound": 100
                },
                "initial-value": 1
            }
        ])");
        for (auto it = variables_json.begin(); it != variables_json.end(); ++it) {
            std::unique_ptr<Variable> var = engine.testConstructVariable(*it);
            engine.testAddVariable(std::move(var));
        }
        // Add some constants to the engine
        nlohmann::json constants_json = nlohmann::json::parse(R"([
            {
                "name": "gravity",
                "type": "real",
                "value": -9.8067
            },
        ])");
        for (auto it = constants_json.begin(); it != constants_json.end(); ++it) {
            std::unique_ptr<Variable> constant = engine.testConstructConstant(*it);
            engine.testAddVariable(std::move(constant));
        }
        // Add some initial states
        nlohmann::json json_obj = nlohmann::json::parse(R"([
            {
                "variables": [
                    {
                        "value": 0,
                        "var": "episode"
                    },
                    {
                        "value": 7.4787,
                        "var": "height"
                    },
                    {
                        "value": -0.651,
                        "var": "velocity"
                    }
                ]
            },
            {
                "variables": [
                    {
                        "value": 0,
                        "var": "episode"
                    },
                    {
                        "value": 8.0739,
                        "var": "height"
                    },
                    {
                        "value": 0.8914,
                        "var": "velocity"
                    }
                ]
            },
            {
                "variables": [
                    {
                        "value": 0,
                        "var": "episode"
                    },
                    {
                        "value": 7.1719,
                        "var": "height"
                    },
                    {
                        "value": 0.2349,
                        "var": "velocity"
                    }
                ]
            }
        ])");
        init_generator = engine.testConstructGeneratorFromValues(json_obj);
    }
}