#ifndef OPENANN_CG_H
#define OPENANN_CG_H

#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/io/Logger.h>
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
  int lineSearchFailed;
public:
  CG()
    : debugLogger(Logger::CONSOLE), opt(0), iteration(-1), n(0),
      SIG(0.1), RHO(0.5*SIG), lineSearchFailed(0)
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
    error = opt->error();
    gradient = opt->gradient();

    double slope = - gradient.transpose() * gradient;
    // initial step is red/(|gradient|+1)
    double x3 = reduction / (1-slope);

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
    parameters.resize(opt->dimension());
    parameters = opt->currentParameters();
    gradient.resize(opt->dimension());
    gradient.fill(0.0);
    iteration = 0;
  }
};

}

#endif // OPENANN_CG_H
