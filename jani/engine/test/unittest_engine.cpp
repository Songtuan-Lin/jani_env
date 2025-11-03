#include <gtest/gtest.h>
#include <random>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include "engine.h"


class AutomatonTest : public ::testing::Test {
protected:
    std::unique_ptr<InitStateGenerator> init_generator;
    std::mt19937 rng;
    void SetUp() override {
        JANIEngine engine;
        rng = std::mt19937(42); // Fixed seed for reproducibility
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
            }
        ])");
        for (auto it = constants_json.begin(); it != constants_json.end(); ++it) {
            std::unique_ptr<Variable> constant = engine.testConstructConstant(*it);
            engine.testAddConstant(std::move(constant));
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
};

struct StateHasher {
    std::size_t operator()(const State& s) const noexcept {
        return std::hash<std::string>{}(s.toString());
    }
};

TEST_F(AutomatonTest, InitialStateGeneration) {
    const int num_samples = 100;
    std::unordered_set<State, StateHasher> samples;
    for (int i = 0; i < num_samples; ++i) {
        State *state = init_generator->generateInitialState(rng);
        samples.insert(*state);
    }
    // Check that all samples are among the defined initial states
    for (const auto& sample : samples) {
        bool found = false;
        for (const auto& init_state : init_generator->getInitialStatePool()) {
            if (sample == (*init_state)) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Sampled state not found in initial states.";
    }

    EXPECT_TRUE(samples.size() > 1); // Ensure variability
    EXPECT_TRUE(samples.size() <= 3); // Ensure we didn't sample too many states
}