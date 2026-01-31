#include "oracle.h"
#include "engine.h"
#include <fstream>
#include "base_components.h"


size_t check_rss_mb() {
    std::ifstream f("/proc/self/statm");
    size_t size, resident;
    f >> size >> resident;
    return (resident * 4) / 1024; // MB (page size = 4KB usually)
}


int main(int argc, char** argv) {
    // std::vector<double> state_vector = {10.0, 15.0, 0.1, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 7.0, 1.0, 1.0, 0.0};
    // std::vector<double> problematic_state_vector = {10.0, 15.0, 0.1, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 8.0, 1.0, 1.0, 0.0};
    // JANIEngine engine(argv[1], argv[2], argv[2], "", "", 42);
    // TarjanOracle oracle(&engine);
    // State prev_state = engine.create_state_from_vector(state_vector);
    // std::cout << "Prev state: \n" << prev_state.toString() << std::endl;
    // State problematic_state = engine.create_state_from_vector(problematic_state_vector);
    // std::cout << "Problematic state: \n" << problematic_state.toString() << std::endl;
    // std::tuple<bool, int> result = oracle.stateSafetyWithAction(prev_state);
    // std::cout << "State is marked " << (std::get<0>(result) ? "safe." : "unsafe.") << " with action " << std::get<1>(result) << std::endl;
    // std::cout << "**************************************************" << std::endl;
    // bool result2 = oracle.isStateSafe(problematic_state);
    // std::cout << "Problematic state is marked " << (result2 ? "safe." : "unsafe.") << std::endl;

    JANIEngine engine(argv[1], argv[2], argv[2], "", "", 42);
    TarjanOracle oracle(&engine, true);
    for (int i = 0; i < 10; i++) {
        engine.reset_with_index(i);
        std::cout << "Memory usage before checking " << i << "th state safety: " << check_rss_mb() << " MB" << std::endl;
        const State& current_state = engine.get_current_state();
        bool result = oracle.isStateSafe(current_state);
        std::cout << "Engine current state is marked " << (result ? "safe." : "unsafe.") << std::endl;
        std::cout << "Memory usage after checking " << i << "th state safety: " << check_rss_mb() << " MB" << std::endl;
    }
    return 0;
}