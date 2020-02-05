#ifndef QUARTZ_RUNGE_KUTTA_H
#define QUARTZ_RUNGE_KUTTA_H

namespace quartz {
namespace math {

template<typename State, typename Potential>
using Runge_Kutta = std::function<
    Propagator<State, Potential>
        (const Propagator<State, Potential> & propagator, const double dt)>;


template<typename State, typename Potential>
Propagator<State, Potential>
    runge_kutta_4(const Propagator<State, Potential> & propagator) {
      
    }


}
}
#endif //QUARTZ_RUNGE_KUTTA_H
