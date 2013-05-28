#include <OpenANN/OpenANN>
#include <OpenANN/optimization/MBSGD.h>
#include "IDXLoader.h"
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/IntrinsicPlasticity.h>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

/**
 * \page MNISTBenchmark MNIST
 *
 * Here, we use a CNN that is similar to Yann LeCun's LeNet 5 to learn
 * handwritten digit  recognition. Download the MNIST data set from
 * <a href="http://yann.lecun.com/exdb/mnist/" target=_blank>THE MNIST
 * DATABASE of handwritten digits</a>. You need all four files. Create the
 * directory "mnist" in your working directory, move the data set to this
 * directory and execute the benchmark or pass the directory of the MNIST
 * data set as argument to the program. Some information about the
 * classification of the test set will be logged in the file "dataset-*.log",
 * where '*' is the starting time.
 *
 * To execute the benchmark you can run the Python script:
\code
python benchmark.py [download] [run] [evaluate]
\endcode
 * download will download the dataset, run will start the benchmark and
 * evaluate will plot the result. You can of course modify the script or do
 * the each step manually.
 */

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif
  std::string directory = "./";
  if(argc > 1)
    directory = std::string(argv[1]);

  IDXLoader loader(28, 28, 60000, 10000, directory);

  OpenANN::Net net;
  net.inputLayer(1, loader.padToX, loader.padToY)
     .extremeLayer(800, OpenANN::LINEAR, 0.5)
     .intrinsicPlasticityLayer(0.2)
     .outputLayer(loader.F, OpenANN::LINEAR, 0.05);
  OpenANN::DirectStorageDataSet trainingSet(&loader.trainingInput, &loader.trainingOutput);
  net.trainingSet(trainingSet);
  OpenANN::DirectStorageDataSet testSet(&loader.testInput, &loader.testOutput,
                                        OpenANN::DirectStorageDataSet::MULTICLASS,
                                        OpenANN::Logger::FILE);
  net.validationSet(testSet);
  net.setErrorFunction(OpenANN::CE);
  OPENANN_INFO << "Created MLP.";
  OPENANN_INFO << "D = " << loader.D << ", F = " << loader.F
               << ", N = " << loader.trainingN << ", L = " << net.dimension();
  OPENANN_INFO << "Press CTRL+C to stop optimization after the next"
      " iteration is finished.";

  {
    OPENANN_INFO << "IP training started.";
    OpenANN::IntrinsicPlasticity& ip = (OpenANN::IntrinsicPlasticity&) net.getLayer(2);
    OpenANN::DataSet* transformedDataSet = net.propagateDataSet(trainingSet, 2);
    ip.trainingSet(*transformedDataSet);
    OpenANN::MBSGD ipOpt(5e-5, 0.9, 1);
    ipOpt.setOptimizable(net);
    OpenANN::StoppingCriteria ipStop;
    ipStop.maximalIterations = 1;
    ipOpt.setStopCriteria(ipStop);
    ipOpt.optimize();
    ip.removeTrainingSet();
    delete transformedDataSet;
    OPENANN_INFO << "IP training finished.";
  }

  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 100;
  OpenANN::MBSGD optimizer(0.01, 0.6, 16, 0.0, 1.0, 0.0, 0.0, 1.0, 0.01, 100.0);
  optimizer.setOptimizable(net);
  optimizer.setStopCriteria(stop);
  optimizer.optimize();

  OPENANN_INFO << "Error = " << net.error();
  OPENANN_INFO << "Wrote data to dataset-*.log.";

  OpenANN::Logger resultLogger(OpenANN::Logger::APPEND_FILE, "weights");
  resultLogger << optimizer.result();

  return EXIT_SUCCESS;
}
