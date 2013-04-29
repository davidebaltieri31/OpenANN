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
  Eigen::VectorXd parameters, error, gradient;
public:
  CG()
    : debugLogger(Logger::CONSOLE), opt(0), iteration(-1), n(0)
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
  }
};

}

#endif // OPENANN_CG_H
