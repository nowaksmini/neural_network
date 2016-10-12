import org.encog.engine.network.activation.ActivationFunction;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.io.File;
import java.util.Arrays;

public class NeuralNetwork {

    private int layers;
    private int iterations;
    private int[] neurons;
    private boolean bias;
    private ActivationFunction activationFunction;
    private double learnRate;
    private double momentum;
    private String[] headers;

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

    private BasicNetwork generateNetwork() {
        BasicNetwork network = new BasicNetwork();

        for (int i = 0; i < layers; i++) {
            network.addLayer(new BasicLayer(activationFunction, bias, neurons[i]));
        }
        network.getStructure().finalizeStructure();
        network.reset();
        return network;
    }

    public void run(String path) {
        BasicNetwork network = generateNetwork();
        File trainingFile = new File(path);

        VersatileDataSource source = new CSVDataSource(trainingFile, true, CSVFormat.DECIMAL_POINT);

        VersatileMLDataSet data = new VersatileMLDataSet(source);
        data.getNormHelper().setFormat(CSVFormat.DECIMAL_POINT);

        ColumnDefinition[] columnDefinitions = new ColumnDefinition[headers.length];

        for (int i = 0; i < columnDefinitions.length; i++) {
            columnDefinitions[i] = data.defineSourceColumn(headers[i], ColumnType.continuous);
        }

        data.analyze();

        for (int i = 0; i < columnDefinitions.length - 1; i++) {
            data.defineInput(columnDefinitions[i]);
        }
        data.defineOutput(columnDefinitions[columnDefinitions.length - 1]);

        EncogModel model = new EncogModel(data);
        model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);
        data.normalize();
        model.holdBackValidation(0.3, false, 1001);

        model.selectTrainingType(data);

        NormalizationHelper helper = data.getNormHelper();

        BasicMLDataSet basicMLDataSet = new BasicMLDataSet();

        for (MLDataPair mlDataPair : model.getDataset()) {
            basicMLDataSet.add(mlDataPair);
        }

        ResilientPropagation trainer = new ResilientPropagation(network, model.getDataset());
//        Backpropagation trainer = new Backpropagation(network, basicMLDataSet, learnRate, momentum);
        System.out.println("Neural Network Iterations:");
        double[] error = new double[iterations];
        int iteration = 0;
        while (iteration < iterations) {
            trainer.iteration();
            error[iteration] = trainer.getError();
            System.out.println("iteration " + iteration + ", error " + error[iteration]);
            iteration++;
        }

        trainer.finishTraining();

        ReadCSV csv = new ReadCSV(trainingFile, true, CSVFormat.DECIMAL_POINT);
        String[] line = new String[headers.length];

        double[] slice = new double[headers.length];

        int index = 0;
        while (csv.next()) {
            StringBuilder result = new StringBuilder();

            for (int i = 0; i < line.length; i++) {
                line[i] = csv.get(i);
            }
            helper.normalizeInputVector(line, slice, true);

            String correct = csv.get(line.length - 1);
            MLData output = network.compute(data.get(index).getInput());
            index++;
            String predicted = helper.denormalizeOutputVectorToString(output)[0];

            result.append(Arrays.toString(line));
            result.append(" -> predicted: ");
            result.append(predicted);
            result.append("(correct: ");
            result.append(correct);
            result.append(")");

            System.out.println(result.toString());
        }
    }
}
