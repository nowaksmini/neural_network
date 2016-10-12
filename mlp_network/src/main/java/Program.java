import org.encog.engine.network.activation.ActivationTANH;

import java.net.URL;

public class Program {

    public static void main(String[] args) {
        Classification classification = new Classification(20, new int[]{2, 10, 1}, true, new ActivationTANH(), 0.6, 0.7);
        String fileName = "data.train.csv";
        URL resource = classification.getClass().getResource(fileName);
        classification.run(resource.getFile());
    }
}
