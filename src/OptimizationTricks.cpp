#include <optimization/OptimizationTricks.h>
#include <cmath>


namespace OpenANN
{

fpt periodicMapping(fpt x, fpt max)
{
  fpt sum = 0.0;
  const fpt f = 1.0/(4.0*max);
  for(int k = 1; k <= 100; k++)
    sum += std::sin(M_PI*k/2.0)*std::sin(2.0*M_PI*k*f*x)/(k*k);
  return std::fabs(8.0*max/(M_PI*M_PI) * sum);
}

}
