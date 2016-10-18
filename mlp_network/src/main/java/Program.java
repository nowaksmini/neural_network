import org.encog.engine.network.activation.ActivationTANH;

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
        NeuralNetwork classification = new NeuralNetwork(2000, new int[]{2, 50, 3}, true, new ActivationTANH(),
                0.05, 0.01, new String[]{"x", "y", "cls"});
        //2, 20, 3, 0.05, 0.01
        //2, 50, 3, 0.05, 0.01
        String fileNameTrainingClassification = "data.train.csv";
        URL resourceTrainingClassification = classification.getClass().getResource(fileNameTrainingClassification);
        classification.train(resourceTrainingClassification.getFile());
        URL resourceTestClassification = classification.getClass().getResource("data.test.csv");
        classification.test(resourceTestClassification.getFile());
    }
}
