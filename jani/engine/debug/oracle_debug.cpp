#include "oracle.h"
#include "engine.h"


int main(int argc, char** argv) {
    std::vector<double> state_vector = {10.0, 15.0, 0.1, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 7.0, 1.0, 1.0, 0.0};
    std::vector<double> problematic_state_vector = {10.0, 15.0, 0.1, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 8.0, 1.0, 1.0, 0.0};
    JANIEngine engine(argv[1], argv[2], argv[2], "", "", 42);
    TarjanOracle oracle(&engine);
    State prev_state = engine.create_state_from_vector(state_vector);
    std::cout << "Prev state: \n" << prev_state.toString() << std::endl;
    State problematic_state = engine.create_state_from_vector(problematic_state_vector);
    std::cout << "Problematic state: \n" << problematic_state.toString() << std::endl;
    std::tuple<bool, int> result = oracle.stateSafetyWithAction(prev_state);
    std::cout << "State is marked " << (std::get<0>(result) ? "safe." : "unsafe.") << " with action " << std::get<1>(result) << std::endl;
    std::cout << "**************************************************" << std::endl;
    bool result2 = oracle.isStateSafe(problematic_state);
    std::cout << "Problematic state is marked " << (result2 ? "safe." : "unsafe.") << std::endl;
    return 0;
}