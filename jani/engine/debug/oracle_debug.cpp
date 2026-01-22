#include "oracle.h"
#include "engine.h"
#include "base_components.h"


int main(int argc, char** argv) {
    JANIEngine engine(argv[1], argv[2], argv[2], "", "", 42);
    TarjanOracle oracle(&engine);
    engine.reset();
    const State& s = engine.get_current_state();
    std::cout << "Initial state: \n" << s.toString() << std::endl;
    std::tuple<bool, int> result = oracle.stateSafetyWithAction(s);
    bool safe = std::get<0>(result);
    int action_id = std::get<1>(result);
    std::cout << "The initial state is " << (safe ? "safe" : "unsafe") << "." << std::endl;
    if (safe) {
        std::cout << "A safe action ID is: " << action_id << std::endl;
    }
    return 0;
}