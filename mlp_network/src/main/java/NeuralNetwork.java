import org.encog.engine.network.activation.ActivationFunction;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class NeuralNetwork {

    private int layers;
    private int iterations;
    private int[] neurons;
    private boolean bias;
    private ActivationFunction activationFunction;
    private double learnRate;
    private double momentum;
    private String[] headers;
    private BasicNetwork network;
    private NormalizationHelper helper;

    public NeuralNetwork(int iterations, int[] neurons, boolean bias, ActivationFunction activationFunction,
                         double learnRate, double momentum, String[] headers) {
        this.layers = neurons.length;
        this.iterations = iterations;
        this.neurons = neurons;
        this.bias = bias;
        this.activationFunction = activationFunction;
        this.learnRate = learnRate;
        this.momentum = momentum;
        this.headers = headers;
    }

    /**
     * Trains and creates neural network.
     *
     * @param path path to training file
     */
    public void train(String path) {
        network = generateNetwork();

        File trainingFile = new File(path);

        VersatileMLDataSet trainingDataSet = readDataSet(trainingFile);

        EncogModel model = new EncogModel(trainingDataSet);
        model.selectMethod(trainingDataSet, MLMethodFactory.TYPE_FEEDFORWARD);

        trainingDataSet.normalize();

        model.holdBackValidation(0.3, false, 1001);
        model.selectTrainingType(trainingDataSet);

        double[][] inputArray = new double[model.getDataset().getData().length][];
        double[][] outputArray = new double[model.getDataset().getData().length][];
        int j = 0;
        for (MLDataPair mlDataPair : model.getDataset()) {
            inputArray[j] = mlDataPair.getInputArray();
            outputArray[j] = mlDataPair.getIdealArray();
            j++;
        }

        final NeuralDataSet trainingSet = new BasicNeuralDataSet(inputArray, outputArray);
        final Backpropagation trainer = new Backpropagation(network, trainingSet, learnRate, momentum);
        trainer.setBatchSize(1);

        System.out.println("Neural Network Training:");
        double[] trainingErrors = new double[iterations];
        int iteration = 0;
        while (iteration < iterations) {
            trainer.iteration();
            trainingErrors[iteration] = trainer.getError();
            System.out.println("iteration " + iteration + ", trainingErrors " + trainingErrors[iteration]);
            iteration++;
        }

        trainer.finishTraining();

        try {
            saveErrorsToFile(trainingErrors);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }

        helper = trainingDataSet.getNormHelper();
        calculateResults(trainingFile, true);
    }

    /**
     * Tests data read from test file.
     *
     * @param path path to tested file
     */
    public void test(String path) {
        File testedFile = new File(path);
        calculateResults(testedFile, false);
    }

    /**
     * Creates BasicNetwork with layers customized with activationFunction, bias appearance and number of neurons on each layer
     *
     * @return BasicNetwork with layers
     */
    private BasicNetwork generateNetwork() {
        BasicNetwork network = new BasicNetwork();

        for (int i = 0; i < layers; i++) {
            network.addLayer(new BasicLayer(activationFunction, bias, neurons[i]));
        }
        network.getStructure().finalizeStructure();
        network.reset();
        return network;
    }

    /**
     * Creates data set from input file. Used for training
     *
     * @param inputFile input csv file
     * @return not normalized data set for model
     */
    private VersatileMLDataSet readDataSet(File inputFile) {
        VersatileDataSource source = new CSVDataSource(inputFile, true, CSVFormat.DECIMAL_POINT);

        VersatileMLDataSet data = new VersatileMLDataSet(source);
        data.getNormHelper().setFormat(CSVFormat.DECIMAL_POINT);

        ColumnDefinition[] columnDefinitions = new ColumnDefinition[headers.length];

        for (int i = 0; i < columnDefinitions.length; i++) {
            if (i == headers.length-1 && headers.length == 3) {
                columnDefinitions[i] = data.defineSourceColumn(headers[i], ColumnType.nominal);
            } else {
                columnDefinitions[i] = data.defineSourceColumn(headers[i], ColumnType.continuous);
            }
        }

        data.analyze();

        for (int i = 0; i < columnDefinitions.length - 1; i++) {
            data.defineInput(columnDefinitions[i]);
        }

        data.defineOutput(columnDefinitions[columnDefinitions.length - 1]);
        return data;
    }

    /**
     * Calculate predicted output and expected. Log results to console and save them to file.
     *
     * @param testedFile tested file - can be also training file
     * @param hasOutput  if input data has output column
     */
    private void calculateResults(File testedFile, boolean hasOutput) {
        System.out.println("Neural network results:");
        ReadCSV csv = new ReadCSV(testedFile, true, CSVFormat.DECIMAL_POINT);
        int length = hasOutput ? headers.length : headers.length - 1;
        String[] line = new String[length];
        double[] lineData = new double[length];
        List<String> inputList = new LinkedList<String>();
        List<String> outputList = new LinkedList<String>();
        List<String> calculatedList = new LinkedList<String>();
        int predictedCount[] = new int[3];
        int expectedCount[] = new int[3];
        while (csv.next()) {
            StringBuilder result = new StringBuilder();
            String inputString = "";
            for (int i = 0; i < line.length; i++) {
                line[i] = csv.get(i);
                lineData[i] = Double.valueOf(line[i]);
                if (i == line.length - 1 && hasOutput) {
                    continue;
                }
                inputString += line[i] + " ";
            }
            helper.normalizeInputVector(line, lineData, true);
            MLData output = network.compute(new BasicMLData(lineData));
            String predicted = helper.denormalizeOutputVectorToString(output)[0];

            result.append(Arrays.toString(line));
            inputList.add(inputString);
            outputList.add(hasOutput ? line[line.length - 1] : "");
            calculatedList.add(predicted);
            result.append(" -> predicted: ");
            result.append(predicted);

//            predictedCount[Integer.parseInt(predicted) - 1]++;
//            expectedCount[Integer.parseInt(line[2]) - 1]++;

            System.out.println(result.toString());
        }

//        System.out.println("Przewidziano wartosci 1, 2, 3 tyle razy: " + predictedCount[0] + ", " +
//        predictedCount[1] + ", " + predictedCount[2]);
//
//        System.out.println("Spodziewano sie wartosci 1, 2, 3 tyle razy: " + expectedCount[0] + ", " +
//                expectedCount[1] + ", " + expectedCount[2]);

        List<List<String>> data = new LinkedList<List<String>>();
        data.add(inputList);
        data.add(outputList);
        data.add(calculatedList);

        try {
            saveResultsToFile(data, hasOutput ? "train.txt" : "test.txt");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves array of errors to file name training-errors@Date.txt
     *
     * @param errors array of errors during training
     * @throws FileNotFoundException
     * @throws UnsupportedEncodingException
     */
    private void saveErrorsToFile(double[] errors) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writerNewest = new PrintWriter("training-erros.txt", "UTF-8");
        for (double error : errors) {
            writerNewest.write(error + "");
        }
        writerNewest.close();
    }

    /**
     * Saves expected and calculated data to file name given
     *
     * @throws FileNotFoundException
     * @throws UnsupportedEncodingException
     */
    private void saveResultsToFile(List<List<String>> data, String fileName)
            throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writerNewest = new PrintWriter(fileName, "UTF-8");
        StringBuilder stringBuilder = new StringBuilder();
        for (List<String> strings : data) {
            for (String string : strings) {
                stringBuilder.append(string).append(" ");
            }
        }
        writerNewest.write(stringBuilder.toString());
        writerNewest.close();
    }

}
