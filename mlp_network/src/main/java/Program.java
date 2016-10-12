import org.encog.engine.network.activation.ActivationTANH;

import java.net.URL;

public class Program {

    public static void main(String[] args) {
        // regresion problem
        NeuralNetwork neuralNetwork = new NeuralNetwork(4000, new int[]{1, 10, 1}, true, new ActivationTANH(),
                0.6, 0.7, new String[]{"x", "y"});
        String fileName = "data.xsq.train.csv";
        URL resource = neuralNetwork.getClass().getResource(fileName);
        neuralNetwork.run(resource.getFile());
        //classification problem
        NeuralNetwork classification = new NeuralNetwork(20, new int[]{2, 10, 1}, true, new ActivationTANH(),
                0.6, 0.7, new String[]{"x", "y", "cls"});
        String fileNameClassification = "data.train.csv";
        URL resourceClassification = classification.getClass().getResource(fileNameClassification);
        classification.run(resourceClassification.getFile());
    }
}
