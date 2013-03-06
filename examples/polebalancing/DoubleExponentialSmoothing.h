#pragma once

#include <Eigen/Dense>
#include <Random.h>

class DoubleExponentialSmoothing
{
  //! Data smoothing factor.
  fpt alpha;
  //! Trend smoothing factor.
  fpt beta;
  //! Current position.
  fpt xc;
  //! Next position.
  fpt xn;
  //! Current position estimation.
  fpt sc;
  //! Next position estimation.
  fpt sn;
  //! Current velocity estimation.
  fpt bc;
  //! Next velocity estimation.
  fpt bn;
  //! Current time step.
  int t;

public:
  DoubleExponentialSmoothing()
    : alpha(0.9), beta(0.9)
  {
    restart();
  }

  void restart()
  {
    xc = 0.0;
    xn = 0.0;
    t = 0;
  }

  Vt operator()(fpt in)
  {
    xc = xn;
    xn = in;
    sc = sn;
    bc = bn;
    if(t == 0)
    {
      sn = in;
      bn = 0.0;
    }
    else if(t == 1)
    {
      sn = xc;
      bn = xn - xc;
    }
    else
    {
      sn = alpha * xn + (1.0-alpha)*(sc + bc);
      bn = beta * (sn - sc) + (1.0-beta) * bc;
    }
    t++;
    Vt out(2);
    out << sn + bn, bn; // Forcast
    return out;
  }

  void initialize()
  {
    OpenANN::RandomNumberGenerator rng;
    alpha = rng.generate<fpt>(0, 1);
    beta = rng.generate<fpt>(0, 1);
  }

  void setParameters(const Vt& parameters)
  {
    alpha = parameters(0);
    beta = parameters(1);
    if(alpha > 1.0)
      alpha = 1.0;
    if(alpha < 0.1)
      alpha = 0.1;
    if(beta > 1.0)
      beta = 1.0;
    if(beta < 0.1)
      beta = 0.1;
  }

  Vt getParameters()
  {
    Vt parameters(2);
    parameters << alpha, beta;
    return parameters;
  }
};
