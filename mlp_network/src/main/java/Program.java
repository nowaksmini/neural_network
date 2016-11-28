import org.encog.engine.network.activation.ActivationTANH;

import java.net.URL;

public class Program {

    public static void main(String[] args) {
        String prePath = "data.multimodal";
        String postPath = "10000.";
        regressionProblem(prePath, postPath);
    }

    private static void classificationProblem(String prePath, String postPath) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(1000, new int[]{2, 10, 10, 4}, false, new ActivationTANH(),
                0.004, 0.001, new String[]{"x", "y", "cls"});
        count(prePath, postPath, neuralNetwork);
    }

    private static void regressionProblem(String prePath, String postPath) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(2000, new int[]{1, 10, 10, 1}, true, new ActivationTANH(),
                0.01, 0.01, new String[]{"x", "y"});
        count(prePath, postPath, neuralNetwork);
    }

    private static void count(String prePath, String postPath, NeuralNetwork neuralNetwork) {
        String fileName = prePath + ".train." + postPath + "csv";
        URL resourceTraining = neuralNetwork.getClass().getResource(fileName);
        neuralNetwork.train(resourceTraining.getFile());
        URL resourceTest = neuralNetwork.getClass().getResource(prePath + ".test." + postPath + "csv");
        neuralNetwork.test(resourceTest.getFile());
    }
}
