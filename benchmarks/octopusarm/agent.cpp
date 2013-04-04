#include "agent.h"
#include <DeepNetwork.h>
#include <optimization/IPOPCMAES.h>
#include <AssertionMacros.h>
#include <EigenWrapper.h>
#include <io/Logger.h>
#include <Random.h>
#include <ctime>
#include <fstream>
#include <stdlib.h>
#include <time.h>

using namespace OpenANN;

/**
 * \page OctopusArm Octopus Arm
 *
 * This is a reinforcement learning problem that has large dimensional state
 * and action space. Both are continuous, thus, we apply neuroevolution to
 * solve this problem.
 *
 * The octopus arm environment is available at <a href=
 * "http://www.cs.mcgill.ca/~dprecup/workshops/ICML06/octopus.html" target=
 * _blank>http://www.cs.mcgill.ca/~dprecup/workshops/ICML06/octopus.html</a>.
 * You have to unpack the archive octopus-code-distribution.zip to the working
 * directory.
 *
 * In this case, we only use the default environment settings. These are very
 * easy to learn. You see it in this picture:
 *
 * \image html octopusarm.png
 *
 * The octopus consists of 12 distinct compartments. It can move 36 muscles
 * and the state space has 106 components. The agent has to move the orange
 * pieces of food into the black mouth. We use an MLP with 106-10-36 topology
 * and bias. The action's components have to be in [0, 1]. Therefore, the
 * activation function of the output layer is logistic sigmoid. In the hidden
 * layer it is tangens hyperbolicus. In this benchmark we compare several
 * compression configurations. The weights of a neuron in the first layer are
 * represented by varying numbers of parameters (5-107) and the weights of a
 * neuron in the second layer are represented by 11 parameters.
 *
 * You can start the benchmark with
\verbatim
ruby run
\endverbatim
 * and evaluate the results with
\verbatim
ruby evaluate
\endverbatim
 * The log files that are needed for the evaluation script are collected in
 * the folder "logs" in the working directory. The evaluation script will list
 * the average return and the maximal return for each run and will calculate
 * the mean and standard deviation of the average returns and the maximum of
 * the maximal returns for each configuration. The average return indicates
 * how fast the agent learns a good policy and the maximal returns indicates
 * how good the best representable policy is.
 *
 * If you run this benchmark on one computer, it takes about 20 days. Thus,
 * it is recommended to start the benchmark on multiple computers. You can
 * modify the variable "runs" in the ruby script "run" and set it to a desired
 * number, start the script on separate computers, merge the results in a
 * single directory "logs" and run the script "evaluate". Each run will take
 * approximately two days. If you set the number of runs to 2 and run the
 * script on 5 computers, it will take about 4 days to finish.
 */

bool fullyObservable = true;
int num_states = 0, num_actions = 0;
int observableStates = 0;
int i = 0;
int behaviors = 6;
double shortestDist;
double xGoal = 11.0; // TODO adjust goal
double yGoal = -6.0;
IPOPCMAES opt;
DeepNetwork net;
double episodeReturn;
Logger logger(Logger::CONSOLE); // TODO log to file
int hiddenUnits;
int parameters;
double bestReturn;
Vt bestParameters;
Vt lastState;

int agent_init(int num_state_variables, int num_action_variables, int argc, const char *agent_param[])
{
  num_states = num_state_variables;
  num_actions = num_action_variables;

  parameters = 0;
  hiddenUnits = 10;
  if(argc > 0)
    parameters = atoi(agent_param[0]);
  if(argc > 1)
    hiddenUnits = atoi(agent_param[1]);
  if(argc > 2)
    fullyObservable = std::string(agent_param[2]) == "mdp";

  if(fullyObservable)
    observableStates = num_states;
  else // remove velocities
    observableStates = (num_states - 2) / 2 + 2;

  net.inputLayer(observableStates);
  if(!fullyObservable)
    net.alphaBetaFilterLayer(0.01);
  if(parameters > 0)
  {
    if(hiddenUnits > 0)
      net.compressedLayer(hiddenUnits, parameters, TANH, "dct");
    net.compressedOutputLayer(behaviors, hiddenUnits+1, LOGISTIC, "dct");
  }
  else
  {
    if(hiddenUnits > 0)
      net.fullyConnectedLayer(hiddenUnits, TANH);
    net.outputLayer(behaviors, LOGISTIC);
  }
  bestParameters = net.currentParameters();
  bestReturn = -std::numeric_limits<double>::max();

  StoppingCriteria stop;
  stop.maximalFunctionEvaluations = 5000;
  stop.maximalRestarts = 1000;
  opt.setOptimizable(net);
  opt.setStopCriteria(stop);
  opt.setSigma0(1.0);
  opt.restart();

  logger << "# " << net.dimension() << " parameters, " << observableStates
      << " state components, " << num_actions << " action components\n"
      << "# " << (fullyObservable ? "MDP" : "POMDP") << "\n\n";
  return 0;
}

const char* agent_get_name()
{
  std::stringstream stream;
  stream << "Neuroevolution_h_" << hiddenUnits << "_p_" << parameters;
  return stream.str().c_str();
}

Vt convert(double state[])
{
  lastState.resize(observableStates);
  if(fullyObservable)
  {
    for(int i = 0; i < num_states; i++)
      lastState(i) = (fpt) state[i];
  }
  else
  {
    lastState(0) = (fpt) state[0];
    lastState(1) = (fpt) state[1];
    for(int i = 0; i < observableStates-2; i+=2)
    {
      lastState(2+i) = state[2+2*i];
      lastState(3+i) = state[3+2*i];
    }
  }
  return lastState;
}

void convert(const Vt& action, double* out)
{
  for(int i = 0; i < num_actions; i++)
    out[i] = (double) action(i);
}

void updateShortestDist()
{
  double xDiff = lastState(lastState.rows() - (fullyObservable ? 4 : 2)) - xGoal;
  double yDiff = lastState(lastState.rows() - (fullyObservable ? 3 : 1)) - yGoal;
  double dist = xDiff*xDiff + yDiff*yDiff;
  if(dist < shortestDist)
    shortestDist = dist;
}

int chooseAction(double state_data[], double out_action[])
{
  Vt state = convert(state_data);
  //logger << "state = " << state.transpose() << "\n";
  OPENANN_CHECK_MATRIX_BROKEN(state);
  Vt y = net(state);
  Vt action(num_actions);
  action.fill(0.0);

  for(int i = 0; i < num_actions/2; i+=3)
    action(i) = y(0);
  for(int i = 1; i < num_actions/2; i+=3)
    action(i) = y(1);
  for(int i = 2; i < num_actions/2; i+=3)
    action(i) = y(2);
  for(int i = num_actions/2+0; i < num_actions; i+=3)
    action(i) = y(3);
  for(int i = num_actions/2+1; i < num_actions; i+=3)
    action(i) = y(4);
  for(int i = num_actions/2+2; i < num_actions; i+=3)
    action(i) = y(5);
  //action *= 10.0;
  convert(action, out_action);
  return 0;
}

int agent_start(double state_data[], double out_action[])
{
  shortestDist = std::numeric_limits<double>::max();
  net.setParameters(opt.getNext());
  episodeReturn = 0;
  chooseAction(state_data, out_action);
  updateShortestDist();
  return 0;
}

int agent_step(double state_data[], double reward, double out_action[])
{
  episodeReturn += reward;
  chooseAction(state_data, out_action);
  updateShortestDist();
  return  0;
}

int agent_end(double reward) {
  episodeReturn += reward;
  if(episodeReturn < 0.0)
    episodeReturn = -shortestDist;
  logger << ++i << " " << episodeReturn << "\n";
  if(episodeReturn > bestReturn)
  {
    bestReturn = episodeReturn;
    bestParameters = net.currentParameters();
  }
  RandomNumberGenerator rng;
  opt.setError(-episodeReturn);
  if(opt.terminated())
    opt.restart();
  return 0;
}

void agent_cleanup()
{
  time_t rawtime;
  struct tm* timeinfo;
  std::time(&rawtime);
  timeinfo = std::localtime(&rawtime);
  std::ofstream result((std::string(std::asctime(timeinfo)).substr(0, 24) + "-best.log").c_str());
  result << "Best Return = " << bestReturn << std::endl;
  result << "Hidden Units = " << hiddenUnits << std::endl;
  result << "Parameters = " << parameters << std::endl;
  result << "Best Parameters = " << std::endl << bestParameters << std::endl;
  result.close();
}
