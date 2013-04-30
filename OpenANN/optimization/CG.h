#ifndef OPENANN_CG_H
#define OPENANN_CG_H

#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/io/Logger.h>
#include <OpenANN/util/EigenWrapper.h>
#include <Eigen/Dense>
#include <limits>

namespace OpenANN {

/**
 * @class CG
 *
 * Conjugate gradient.
 *
 * This implementation is based on Carl Edward Rasmussen's Matlab
 * implementation.
 */
class CG : public Optimizer
{
  Logger debugLogger;
  StoppingCriteria stop;
  Optimizable* opt; // do not delete
  Eigen::VectorXd optimum;
  int iteration, n;
  double error;
  Eigen::VectorXd parameters, gradient;

  /**
   * SIG and RHO are the constants controlling the Wolfe-Powell conditions.
   * SIG is the maximum allowed absolute ratio between previous and new slopes
   * (derivatives in the search direction), thus setting SIG to low (positive)
   * values forces higher precision in the line-searches. Constants must
   * satisfy 0 < RHO < SIG < 1. Tuning of SIG (depending on the nature of the
   * function to be optimized) may speed up the minimization; it is probably
   * not worth playing much with RHO.
   */
  const double SIG;
  /**
   * RHO is the minimum allowed fraction of the expected (from the slope at
   * the initial point in the linesearch).
   */
  const double RHO;
  /**
   * Maximum of function evaluations per line search.
   */
  const int MAX_LINE_SEARCH;
  /**
   * Don't reevaluate within INT of the limit of the current bracket.
   */
  const double INT;
  /**
   * Maximum extrapolation factor of current step-size.
   */
  const double EXT;
  /**
   * Maximum allowed slope ratio.
   */
  const double RATIO;
  int lineSearchFailed;
  Eigen::VectorXd s, X, df0, X0, dF0, df3;
  double fX, F0,
      f0, d0,
      x1, f1, d1,
      x2, f2, d2,
      x3, f3, d3,
      x4, f4, d4;
public:
  CG()
    : debugLogger(Logger::CONSOLE), opt(0), iteration(-1), n(0),
      SIG(0.1), RHO(0.5*SIG), MAX_LINE_SEARCH(20), INT(0.1), EXT(3.0),
      RATIO(10.0), lineSearchFailed(0)
  {
  }

  virtual ~CG()
  {
  }

  virtual void setOptimizable(Optimizable& opt)
  {
    this->opt = &opt;
  }

  virtual void setStopCriteria(const StoppingCriteria& stop)
  {
    this->stop = stop;
  }

  virtual void optimize()
  {
    if(iteration < 0)
      initialize();

    while(step());
  }

  virtual bool step()
  {
    bool abort = false;
    double red = 1.0; // reduction in function value to be expected in the first line-search, TODO
    // TODO mini-batch cg
    X = opt->currentParameters();
    f0 = opt->error();
    df0 = opt->gradient();
    fX = f0;
    OPENANN_INFO << 0 << " " << f0;

    s = -gradient; // initial search direction (steepest)
    d0 = -s.transpose() * s; // slope
    x3 = red / (1-d0); // initial step is red/(|s|+1)

    X0 = X;
    F0 = f0;
    dF0 = df0;

    int M = MAX_LINE_SEARCH;

    while(true) // keep extrapolating as long as necessary
    {
      x2 = 0.0;
      f2 = f0;
      d2 = d0;
      f3 = f0;
      df3 = df0;

      bool success = false;
      do {
        M--;
        opt->setParameters(X + x3*s);
        f3 = opt->error();
        OPENANN_INFO << x3 << " " << f3;
        df3 = opt->gradient();
        if(isnan(f3) || isinf(f3) || isMatrixBroken(df3))
          x3 = (x2+x3)/2; // bisect and try again
        else
          success = true;
      } while(!success && M > 0);

      // keep best values
      if(f3 < F0)
      {
        X0 = X + x3*s;
        F0 = f3;
        dF0 = df3;
      }
      d3 = df3.transpose() * s; // new slope

      // are we done extrapolating?
      if(d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0)
        break;
    }

    // move point 2 to point 1
    x1 = x2;
    f1 = f2;
    d1 = d2;
    // move point 3 to point 2
    x2 = x3;
    f2 = f3;
    d2 = d3;
    // make cubic extrapolation
    double diffx2x1 = x2-x1;
    double A = 6*(f1-f2)+3*(d2+d1)*diffx2x1;
    double B = 3*(f2-f1)-(2*d1+d2)*diffx2x1;
    x3 = x1-d1*diffx2x1*diffx2x1/(B+std::sqrt(B*B-A*d1*(x2-x1))); // num. error possible, ok!
    if(isnan(x3) || isinf(x3) || x3 < 0) // num prob | wrong sign?
      x3 = x2*EXT; // extrapolate maximum amount
    else if(x3 > x2*EXT) // new point beyond extrapolation limit?
      x3 = x2*EXT; // extrapolate maximum amount
    else if(x3 < x2+INT*(x2-x1)) // new point too close to previous point?
      x3 = x2+INT*(x2-x1);

    while((std::abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0) // keep interpolating
    {
      if(d3 > 0 || f3 > f0+x3*RHO*d0) // choose subinterval
      { // move point 3 to point 4
        x4 = x3;
        f4 = f3;
        d4 = d3;
      }
      else
      { // move point 3 to point 2
        x2 = x3;
        f2 = f3;
        d2 = d3;
      }
      double diffx4x2 = x4-x2;
      if(f4 > f0)
      { // quadratic interpolation
        x3 = x2-(0.5*d2*diffx4x2*diffx4x2)/(f4-f2-d2*diffx4x2);
      }
      else
      { // cubic interpolation
        double A = 6*(f2-f4)/diffx4x2+3*(d4+d2);
        double B = 3*(f4-f2)-(2*d2+d4)*diffx4x2;
        x3 = x2+(sqrt(B*B-A*d2*diffx4x2*diffx4x2)-B)/A; // num. error possible, ok!
      }
      if(std::isnan(x3) || std::isinf(x3))
        x3 = (x2+x4)/2; // if we had a numerical problem then bisect
      x3 = std::max(std::min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2)); // don't accept too close
      opt->setParameters(X+x3*s);
      f3 = opt->error();
      OPENANN_INFO << x3 << " " << f3;
      df3 = opt->gradient();
      if(f3 < F0)
      { // keep best values
        X0 = X + x3*s;
        F0 = f3;
        dF0 = df3;
      }
      M--;
      d3 = df3.transpose()*s; // new slope
    }

    if(std::abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0) // if line search succeeded
    {
      // update variables
      X = X+x3*s;
      f0 = f3;

      // Polack-Ribiere CG direction
      s = (df3.transpose()*df3-df0.transpose()*df3)/(df0.transpose()*df0)*s - df3;
      df0 = df3; // swap derivatives
      d3 = d0;
      d0 = df0.transpose()*s;
      if(d0 > 0) // new slope must be negative
      { // otherwise use steepest direction
        s = -df0;
        d0 = -s.transpose()*s;
      }
      x3 = x3 * std::min(RATIO, d3/(d0-std::numeric_limits<double>::min())); // slope ratio but max RATIO
      lineSearchFailed = 0; // this line search did not fail
    }
    else
    {
      // restore best point so far
      X = X0;
      f0 = F0;
      df0 = dF0;

      if(lineSearchFailed) // line search failed twice in a row
        abort = true;
      else
      {
        s = -df0;
        d0 = -s.transpose()*s; // try steepest
        x3 = 1/(1-d0);
        lineSearchFailed = 1; // this line search failed
      }
    }

    iteration++;
    bool run = !abort &&
        (stop.maximalIterations != stop.defaultValue.maximalIterations
         && iteration >= stop.maximalIterations);
    if(!run)
      iteration = -1;
    return run;
  }

  virtual Eigen::VectorXd result()
  {
    return optimum;
  }
  virtual std::string name()
  {
    return "Conjugate Gradient";
  }
private:
  void initialize()
  {
    n = opt->dimension();
    parameters.resize(n);
    parameters = opt->currentParameters();
    gradient.resize(n);
    gradient.fill(0.0);
    iteration = 0;
  }
};

}

#endif // OPENANN_CG_H
