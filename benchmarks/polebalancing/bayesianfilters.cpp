#include <OpenANN>
#include <SinglePoleBalancing.h>
#include <DoublePoleBalancing.h>
#include <NeuroEvolutionAgent.h>
#include <Test/Stopwatch.h>
#include <numeric>
#include <vector>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

/**
 * \page BayesianFilterBenchmark Bayes Filters
 *
 * This benchmark is based on the example \ref PB.
 *
 * Open question:
 *
 *  - Is the noise system inherent noise or just measurement noise?
 *
 */

struct Result
{
  bool success;
  unsigned long episodes;
  unsigned long maxSteps;
  unsigned long time;
};

struct Results
{
  int runs;
  int failures;
  int median, min, max;
  fpt mean, stdDev, time;
  fpt meanSteps, stdDevSteps;
  Results()
    : runs(0), failures(0), median(0), min(0), max(0), mean(0), stdDev(0),
      time(0), meanSteps(0), stdDevSteps(0)
  {
  }
};

Result benchmarkSingleRun(OpenANN::Environment& environment, OpenANN::Agent& agent)
{
  Result result;
  int maximalEpisodes = 100000;
  int requiredSteps = 100000;
  agent.abandoneIn(environment);

  result.success = false;
  result.episodes = maximalEpisodes;
  Stopwatch sw;
  for(int i = 1; i <= maximalEpisodes; i++)
  {
    environment.restart();
    while(!environment.terminalState())
      agent.chooseAction();
    if(environment.stepsInEpisode() >= requiredSteps)
    {
      result.success = true;
      result.episodes = i;
      break;
    }
  }
  result.maxSteps = environment.stepsInEpisode();
  result.time = sw.stop(Stopwatch::MILLISECOND);

  return result;
}

Results benchmarkConfiguration(bool doublePole, bool fullyObservable,
    bool alphaBetaFilter, bool doubleExponentialSmoothing,
    bool learnDESParameters, int parameters, int runs, fpt sigma0, fpt noise)
{
  OpenANN::Environment* env;
  if(doublePole)
    env = new DoublePoleBalancing(fullyObservable, noise);
  else
    env = new SinglePoleBalancing(fullyObservable, noise);

  Results results;
  results.runs = runs;
  std::vector<fpt> episodes;
  std::vector<fpt> steps;

  OpenANN::Logger progressLogger(Logger::CONSOLE);
  for(int run = 0; run < runs; run++)
  {
    NeuroEvolutionAgent agent(0, false, "linear", parameters > 0, parameters,
        fullyObservable, alphaBetaFilter, doubleExponentialSmoothing,
        learnDESParameters);
    agent.setSigma0(sigma0);
    Result result = benchmarkSingleRun(*env, agent);
    if(run % 10 == 9)
      progressLogger << ".";
    if(!result.success)
      results.failures++;
    episodes.push_back(result.episodes);
    steps.push_back(result.maxSteps);
    results.time += result.time;
    results.mean += result.episodes;
    results.meanSteps += result.maxSteps;
  }
  progressLogger << "\n";
  results.time /= (fpt) runs;
  results.mean /= (fpt) runs;
  results.meanSteps /= (fpt) runs;
  results.min = (int) *std::min_element(episodes.begin(), episodes.end());
  results.max = (int) *std::max_element(episodes.begin(), episodes.end());
  std::sort(episodes.begin(), episodes.end());
  results.median = (int) episodes[episodes.size()/2];
  for(int run = 0; run < runs; run++)
  {
    episodes[run] -= results.mean;
    episodes[run] *= episodes[run];
    steps[run] -= results.meanSteps;
    steps[run] *= steps[run];
  }
  results.stdDev = std::sqrt(std::accumulate(episodes.begin(), episodes.end(), (fpt) 0) / (fpt) runs);
  results.stdDevSteps = std::sqrt(std::accumulate(steps.begin(), steps.end(), (fpt) 0) / (fpt) runs);

  delete env;
  return results;
}

void printResults(const Results& results)
{
  typedef OpenANN::FloatingPointFormatter fmt;
  OpenANN::Logger resultLogger(OpenANN::Logger::CONSOLE);
  resultLogger << results.failures << "/" << results.runs
      << " failed\nepisodes:\t" << fmt(results.mean, 3) << "+-"
      << fmt(results.stdDev, 4) << "\nrange:\t\t[" << results.min << ","
      << results.max << "]\nmedian:\t\t" << results.median << "\ntime:\t\t"
      << results.time << " ms\nsteps:\t\t" << results.meanSteps << "+-"
      << results.stdDevSteps << "\n\n";
}

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif

  OpenANN::Logger configLogger(OpenANN::Logger::CONSOLE);
  int runs = 10;

  Results results;

  for(fpt noiseLevel = 1.0; noiseLevel <= 3.0; noiseLevel += 0.5)
  {
    fpt noise = noiseLevel * 10.0 / 1024.0;
    configLogger << "=== Noise: " << noise << " (" << noiseLevel << ") ===\n";
    configLogger << "SPB, POMDP (Diff), uncompressed\n";
    results = benchmarkConfiguration(false, false, false, false, false, -1, runs, 10.0, noise);
    printResults(results);
    configLogger << "SPB, POMDP (ABF), uncompressed\n";
    results = benchmarkConfiguration(false, false, true, false, false, -1, runs, 10.0, noise);
    printResults(results);
    configLogger << "SPB, POMDP (DES), uncompressed\n";
    results = benchmarkConfiguration(false, false, false, true, false, -1, runs, 10.0, noise);
    printResults(results);
    configLogger << "SPB, POMDP (DESO), uncompressed\n";
    results = benchmarkConfiguration(false, false, false, true, true, -1, runs, 10.0, noise);
    printResults(results);
    configLogger << "SPB, POMDP (ABF), compressed (3)\n";
    results = benchmarkConfiguration(false, false, true, false, false, 3, runs, 10.0, noise);
    printResults(results);
    configLogger << "SPB, POMDP (DES), compressed (3)\n";
    results = benchmarkConfiguration(false, false, false, true, false, 3, runs, 10.0, noise);
    printResults(results);
    configLogger << "SPB, POMDP (DESO), compressed (3)\n";
    results = benchmarkConfiguration(false, false, false, true, true, 3, runs, 10.0, noise);
    printResults(results);
  }

  configLogger << "=== Without noise ===\n";
  configLogger << "SPB, POMDP (Diff), uncompressed\n";
  results = benchmarkConfiguration(false, false, false, false, false, -1, runs, 10.0, 0);
  printResults(results);
  configLogger << "SPB, POMDP (ABF), uncompressed\n";
  results = benchmarkConfiguration(false, false, true, false, false, -1, runs, 10.0, 0);
  printResults(results);
  configLogger << "SPB, POMDP (DES), uncompressed\n";
  results = benchmarkConfiguration(false, false, false, true, false, -1, runs, 10.0, 0);
  printResults(results);
  configLogger << "SPB, POMDP (DESO), uncompressed\n";
  results = benchmarkConfiguration(false, false, false, true, true, -1, runs, 10.0, 0);
  printResults(results);
  configLogger << "DPB, POMDP (Diff), uncompressed\n";
  results = benchmarkConfiguration(true, false, false, false, false, -1, runs, 10.0, 0);
  printResults(results);
  configLogger << "DPB, POMDP (ABF), uncompressed\n";
  results = benchmarkConfiguration(true, false, true, false, false, -1, runs, 10.0, 0);
  printResults(results);
  configLogger << "DPB, POMDP (DES), uncompressed\n";
  results = benchmarkConfiguration(true, false, false, true, false, -1, runs, 10.0, 0);
  printResults(results);
  configLogger << "DPB, POMDP (DESO), uncompressed\n";
  results = benchmarkConfiguration(true, false, false, true, true, -1, runs, 10.0, 0);
  printResults(results);
  configLogger << "SPB, POMDP (ABF), compressed (3)\n";
  results = benchmarkConfiguration(false, false, true, false, false, 3, runs, 10.0, 0);
  printResults(results);
  configLogger << "SPB, POMDP (DES), compressed (3)\n";
  results = benchmarkConfiguration(false, false, false, true, false, 3, runs, 10.0, 0);
  printResults(results);
  configLogger << "SPB, POMDP (DESO), compressed (3)\n";
  results = benchmarkConfiguration(false, false, false, true, true, 3, runs, 10.0, 0);
  printResults(results);
  configLogger << "DPB, POMDP (ABF), compressed (5)\n";
  results = benchmarkConfiguration(true, false, true, false, false, 5, runs, 10.0, 0);
  printResults(results);
  configLogger << "DPB, POMDP (DES), compressed (5)\n";
  results = benchmarkConfiguration(true, false, false, true, false, 5, runs, 10.0, 0);
  printResults(results);
  configLogger << "DPB, POMDP (DESO), compressed (5)\n";
  results = benchmarkConfiguration(true, false, false, true, true, 5, runs, 10.0, 0);
  printResults(results);
}
