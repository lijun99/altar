#include "wright_omega.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << std::setprecision(12);

    // omega(0) = W0(1) ≈ 0.5671432904097839
    std::cout << wright_omega(0.0)  << "\n";  // 0.567143290409784
    std::cout << wright_omega(1.0)  << "\n";  // 1.0
    std::cout << wright_omega(-1.0) << "\n";  // 0.278464542761073
    std::cout << wright_omega(100.0) << "\n"; // ≈ 95.6288...
    std::cout << wright_omega(599377.5008473693) << "\n"; // ≈ 599364.1972226682
    std::cout << wright_omega(-22.49915263070216) << "\n"; // ≈ 1.6933321922754353e-10

    // Single precision
    std::cout << wright_omega(0.0f)  << "\n";
    std::cout << wright_omega(1.0f)  << "\n";
    std::cout << wright_omega(-1.0f) << "\n";
    std::cout << wright_omega(100.0f) << "\n";
    std::cout << wright_omega(599377.5008473693f) << "\n";
    std::cout << wright_omega(-22.49915263070216f) << "\n"; 

    // Edge cases
    std::cout << wright_omega(std::numeric_limits<double>::infinity()) << "\n"; // inf
    std::cout << wright_omega(-std::numeric_limits<double>::infinity()) << "\n"; // 0
    std::cout << wright_omega(std::nan("")) << "\n"; // nan
    return 0;
}
