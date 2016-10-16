import org.encog.engine.network.activation.ActivationFunction;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
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
    private VersatileMLDataSet trainingDataSet;

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

        trainingDataSet = readDataSet(trainingFile, true);

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

        calculateResults(trainingFile, trainingDataSet, true);
    }

    /**
     * Tests date read from test file.
     *
     * @param path path to tested file
     */
    public void test(String path) {
        File testedFile = new File(path);
        VersatileMLDataSet testedDataSet = readDataSet(testedFile, false);
        EncogModel model = new EncogModel(testedDataSet);
        model.selectMethod(testedDataSet, MLMethodFactory.TYPE_FEEDFORWARD);
        testedDataSet.normalize();
        model.holdBackValidation(0.3, false, 1001);
        model.selectTrainingType(trainingDataSet);
        calculateResults(testedFile, model.getDataset(), false);
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
     * Creates data set from input file. Used for training and testing
     *
     * @param inputFile input csv file
     * @param hasOutput if true define output column for training
     * @return not normalized data set for model
     */
    private VersatileMLDataSet readDataSet(File inputFile, boolean hasOutput) {
        VersatileDataSource source = new CSVDataSource(inputFile, true, CSVFormat.DECIMAL_POINT);

        VersatileMLDataSet data = new VersatileMLDataSet(source);
        data.getNormHelper().setFormat(CSVFormat.DECIMAL_POINT);

        ColumnDefinition[] columnDefinitions = new ColumnDefinition[headers.length];

        for (int i = 0; i < columnDefinitions.length; i++) {
            if (!hasOutput && (i == columnDefinitions.length - 1)) {
                break;
            }
            columnDefinitions[i] = data.defineSourceColumn(headers[i], ColumnType.continuous);
        }

        data.analyze();

        for (int i = 0; i < columnDefinitions.length - 1; i++) {
            data.defineInput(columnDefinitions[i]);
        }
        if (hasOutput) {
            data.defineOutput(columnDefinitions[columnDefinitions.length - 1]);
        }

        return data;
    }

    /**
     * Calculate predicted output and expected. Log results to console and save them to file.
     *
     * @param testedFile tested file - can be also training file
     * @param inputData  data from encog model, already normalized
     * @param hasOutput  if input data has output column
     */
    private void calculateResults(File testedFile, VersatileMLDataSet inputData, boolean hasOutput) {
        System.out.println("Neural network results:");
        ReadCSV csv = new ReadCSV(testedFile, true, CSVFormat.DECIMAL_POINT);
        if (hasOutput) {
            helper = inputData.getNormHelper();
        }
        int length = hasOutput ? headers.length : headers.length - 1;
        String[] line = new String[length];
        int index = 0;
        List<String> inputList = new LinkedList<String>();
        List<String> outputList = new LinkedList<String>();
        List<String> calculatedList = new LinkedList<String>();
        while (csv.next()) {
            StringBuilder result = new StringBuilder();
            String inputString = "";
            for (int i = 0; i < line.length; i++) {
                line[i] = csv.get(i);
                if (i == line.length - 1 && hasOutput) {
                    continue;
                }
                inputString += line[i] + " ";
            }
            MLData output = network.compute(inputData.get(index).getInput());
            String predicted = helper.denormalizeOutputVectorToString(output)[0];

            result.append(Arrays.toString(line));
            inputList.add(inputString);
            outputList.add(hasOutput ? line[line.length - 1] : "");
            calculatedList.add(predicted);
            result.append(" -> predicted: ");
            result.append(predicted);

            System.out.println(result.toString());
            index++;
        }

        if (headers.length == 2) {
            try {
                saveRegressionToFile(inputList, outputList, calculatedList, hasOutput ? "regression_test.txt" : "regression.txt");
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        // todo save to file
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
     * Saves expected and calculated data of regression problem to file name given
     *
     * @throws FileNotFoundException
     * @throws UnsupportedEncodingException
     */
    private void saveRegressionToFile(List<String> x, List<String> y, List<String> calculated, String fileName)
            throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writerNewest = new PrintWriter(fileName, "UTF-8");
        StringBuilder stringX = new StringBuilder();
        StringBuilder stringY = new StringBuilder();
        StringBuilder stringCalculated = new StringBuilder();
        for (String _x : x) {
            stringX.append(_x).append(" ");
        }
        for (String _y : y) {
            stringY.append(_y).append(" ");
        }
        for (String _calculated : calculated) {
            stringCalculated.append(_calculated).append(" ");
        }
        writerNewest.write(stringX.toString());
        writerNewest.write(stringY.toString());
        writerNewest.write(stringCalculated.toString());
        writerNewest.close();
    }

}
