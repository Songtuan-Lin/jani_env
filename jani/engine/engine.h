#include <vector>
#include <string>
#include <unordered_map>
#include <base_components.h>


class JANIEngine {
    std::vector<Action*> actions;
    std::vector<Variable*> constants;
    std::vector<Variable*> variables;
public:
    ~JANIEngine() {
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