import org.encog.engine.network.activation.ActivationBiPolar;

import java.net.URL;

public class Program {

    public static void main(String[] args) {
        // regression problem
//        NeuralNetwork neuralNetwork = new NeuralNetwork(2000, new int[]{1, 10, 10, 1}, true, new ActivationTANH(),
//                0.01, 0.01, new String[]{"x", "y"});
//        String fileName = "data.xsq.train.csv";
//        URL resource = neuralNetwork.getClass().getResource(fileName);
//        neuralNetwork.train(resource.getFile());
//        URL resourceTest = neuralNetwork.getClass().getResource("data.xsq.test.csv");
//        neuralNetwork.test(resourceTest.getFile());
        //classification problem
        NeuralNetwork classification = new NeuralNetwork(5000, new int[]{2, 20, 30, 3}, true, new ActivationBiPolar(),
                0.15, 0.41, new String[]{"x", "y", "cls"});
        String fileNameTrainingClassification = "data.train.csv";
        URL resourceTrainingClassification = classification.getClass().getResource(fileNameTrainingClassification);
        classification.train(resourceTrainingClassification.getFile());
        URL resourceTestClassification = classification.getClass().getResource("data.test.csv");
        classification.test(resourceTestClassification.getFile());
    }
}
