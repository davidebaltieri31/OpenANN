#ifndef OPENANN_CG_H
#define OPENANN_CG_H

#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/io/Logger.h>
#include <OpenANN/util/EigenWrapper.h>
#include <Eigen/Dense>

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
  int lineSearchFailed;
public:
  CG()
    : debugLogger(Logger::CONSOLE), opt(0), iteration(-1), n(0),
      SIG(0.1), RHO(0.5*SIG), MAX_LINE_SEARCH(20), lineSearchFailed(0)
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
    double reduction = 1.0; // reduction in function value to be expected in the first line-search, TODO
    // 1) Compute error / gradient
    // TODO mini-batch cg
    Eigen::VectorXd parameters = opt->currentParameters();
    this->parameters = parameters;
    double error = opt->error();
    this->error = error;
    Eigen::VectorXd gradient = opt->gradient();
    this->gradient = gradient;

    double slope = - gradient.transpose() * gradient;
    double x1, error1, slope1;
    double x2, error2, slope2;
    // initial step is red/(|gradient|+1)
    double x3 = reduction / (1-slope), error3, slope3;
    Eigen::VectorXd gradient3(n);
    double x0 = 0, error0 = error, slope0 = slope;

    while(true) // keep extrapolating as long as necessary
    {
      x2 = 0.0;
      error2 = error0;
      slope2 = slope0;
      error3 = error0;
      slope3 = slope0;

      int m = MAX_LINE_SEARCH;
      bool success = false;
      double error3;
      Eigen::VectorXd gradient3(n);
      do {
        m--;
        opt->setParameters(parameters - x3*gradient);
        error3 = opt->error();
        gradient3 = opt->gradient();
        if(isnan(error3) || isinf(error3) || isMatrixBroken(gradient3))
          x3 = (x2+x3)/2; // bisect and try again
        else
          success = true;
      } while(!success && m > 0);

      // keep best values
      if(error3 < error)
      {
        parameters = parameters - x3*gradient;
        error = error3;
        gradient = gradient3;
      }
      slope3 = gradient3.transpose() * (-this->gradient); // new slope

      // are we done extrapolating?
      if(slope3 > SIG*slope || error3 > this->error+x3*RHO*slope)
        break;
    }

    // move point 2 to point 1
    x1 = x2;
    error1 = error2;
    gradient1 = gradient2;
    // move point 3 to point 2
    x2 = x3;
    error2 = error3;
    gradient2 = gradient3;
    // make cubic extrapolation
    double a = 6*(error1-error2)+3*(gradient2+gradient1)*(x2-x1);
    // TODO implement
    iteration++;
    bool run = (stop.maximalIterations != stop.defaultValue.maximalIterations
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
