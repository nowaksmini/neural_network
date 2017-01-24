package deeplearning;

import html.Program;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;


public class TranslatorProgram {
    private static final String NETWORK_PATH = "network_many_10000_12_1hidden.txt";
    public static final String TRAINING_SET_PATH = "translate.txt";
    private static final String TEST_SET_PATH = "test.txt";

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Select mode:");
        System.out.println("1 Train neural network");
        System.out.println("2 Test existing neural network with own words");
        System.out.println("3 Test existing neural network with prepared test data");
        System.out.println("Your choice: ");
        int mode = scanner.nextInt();
        if (mode == 1) {
            System.out.println();
            System.out.println("Started training neural network");
            trainNetwork();
        } else if (mode == 2) {
            System.out.println();
            System.out.print("Enter numbers of tests: ");
            int tests = scanner.nextInt();
            ComputationGraph net = readTrainedNeuralNetwork();
            for (int i = 0; i < tests; i++) {
                System.out.print("Enter text in english to translate (max " + TranslatorIterator.MAX_LINE_LENGTH + ") signs: ");
                scanner.nextLine();
                String englishWord = scanner.nextLine();
                testNetwork(net, englishWord, "?");
            }
        } else {
            ComputationGraph net = readTrainedNeuralNetwork();
            try {
                ClassLoader classLoader = Program.class.getClassLoader();
                FileReader file = new FileReader(classLoader.getResource(TEST_SET_PATH).getFile());

                BufferedReader bw = new BufferedReader(file);
                String line;
                while ((line = bw.readLine()) != null) {
                    testNetwork(net, line, "?");
                }
                bw.close();
                file.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static ComputationGraph readTrainedNeuralNetwork() {
        ComputationGraph net;
        try {
            FileInputStream file = new FileInputStream(NETWORK_PATH);
            ObjectInputStream objectOutputStream = new ObjectInputStream(file);
            net = ModelSerializer.restoreComputationGraph(objectOutputStream);
            objectOutputStream.close();
            file.close();
            return net;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private static void trainNetwork() {
        int miniBatchSize = 1; // Number of file lines used when training
        int numHiddenNodes = 100;
        int tbpttLength = 10; //Length for truncated backpropagation through time. i.e., do parameter updates ever 52 characters
        String[] validCharacters = getMinimalCharacterSet();
        int FEATURE_VEC_SIZE = validCharacters.length * TranslatorIterator.MAX_LINE_LENGTH;
        int seed = 123467;
        TranslatorIterator translatorIterator = new TranslatorIterator(seed, miniBatchSize, 10, validCharacters);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.01)
                .rmsDecay(0.95)
                .regularization(true)
                .l2(0.001) // regularization coefficient Use with .regularization(true)
                .updater(Updater.RMSPROP)  //Gradient updater.
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .seed(seed)
                .graphBuilder()
                .addInputs("translateInput")
                .setInputTypes(InputType.recurrent(FEATURE_VEC_SIZE))
                .addLayer("encoderLayer",
                        new GravesLSTM.Builder()
                                .nIn(FEATURE_VEC_SIZE)
                                .nOut(numHiddenNodes)
                                .activation("tanh")
                                .build(),
                        "translateInput")
                .addLayer("hiddenLayer1",
                        new GravesLSTM.Builder()
                                .nIn(numHiddenNodes)
                                .nOut(numHiddenNodes)
                                .activation("tanh")
                                .build(),
                        "encoderLayer")
                .addLayer("outputLayer",
                        new RnnOutputLayer.Builder()
                                .nIn(numHiddenNodes)
                                .nOut(FEATURE_VEC_SIZE)
                                .activation("softmax")
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .build(),
                        "hiddenLayer1")
                .setOutputs("outputLayer")
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(tbpttLength)
                //When doing truncated BPTT: how many steps of forward pass should we do before doing (truncated) backprop?
                //Only applicable when doing backpropType(BackpropType.TruncatedBPTT)
                //Typically tBPTTForwardLength parameter is same as the tBPTTBackwardLength parameter, but may be larger than it in some circumstances (but never smaller)
                //Ideally your training data time series length should be divisible by this This is the k1 parameter on pg23 of http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
                .tBPTTBackwardLength(tbpttLength) //When doing truncated BPTT: how many steps of backward should we do?
                //Only applicable when doing backpropType(BackpropType.TruncatedBPTT)
                //This is the k2 parameter on pg23 of http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
                .pretrain(false) //Whether to do layerwise pre training or not
                .backprop(true)
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        //Train model:
        int iEpoch = 0;
        while (iEpoch < 2000) {
            while (translatorIterator.hasNext()) {
                DataSet ds = translatorIterator.next();
                MultiDataSet multiDataSet = new MultiDataSet(ds.getFeatures(), ds.getLabels());
                net.fit(multiDataSet);
                INDArray[] output = net.output(ds.getFeatures());
                List<String> englishSelected = translatorIterator.getEnglishSelected();
                List<String> polishSelected = translatorIterator.getPolishSelected();
                INDArray predictions = output[0];
                INDArray answers = Nd4j.argMax(predictions, 1);
                checkAnswers(englishSelected, polishSelected, answers, translatorIterator.getValidCharacters());
            }
            iEpoch++;
            translatorIterator.reset();
            System.out.println("\n* = * = * = * = * = * = * = * = * = ** EPOCH " + iEpoch + " COMPLETE ** = * = * = * = * = * = * = * = * = * = * = * = * = * =\n");
        }

        saveNeuralNetwork(net);

        System.out.println("\n\nNetwork saved");
    }

    private static void testNetwork(ComputationGraph net, String englishInput, String polishInput) {
        String[] validCharacters = getMinimalCharacterSet();
        TranslatorIterator translatorIterator = new TranslatorIterator(123467, 1, 1, validCharacters,
                Collections.singletonList(englishInput), Collections.singletonList(polishInput));
        while (translatorIterator.hasNext()) {
            DataSet ds = translatorIterator.next(1);
            INDArray[] output = net.output(ds.getFeatures());
            List<String> englishSelected = translatorIterator.getEnglishSelected();
            List<String> polishSelected = translatorIterator.getPolishSelected();
            INDArray predictions = output[0];
            INDArray answers = Nd4j.argMax(predictions, 1);
            checkAnswers(englishSelected, polishSelected, answers, translatorIterator.getValidCharacters());
        }
    }

    private static void saveNeuralNetwork(ComputationGraph computationGraph) {
        try {
            FileOutputStream file = new FileOutputStream(NETWORK_PATH);
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(file);
            ModelSerializer.writeModel(computationGraph, objectOutputStream, true);
            objectOutputStream.close();
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void checkAnswers(List<String> english, List<String> polish, INDArray answers,
                                     String[] validCharacters) {
        int nTests = answers.size(0);
        for (int iTest = 0; iTest < nTests; iTest++) {
            int aDigit = 0;
            String strAnswer = "";
            while (aDigit < TranslatorIterator.MAX_LINE_LENGTH) {
                int thisDigit = answers.getInt(iTest, aDigit);
                if (thisDigit > validCharacters.length) {
                    thisDigit = 0; // SPACE
                }
                strAnswer = strAnswer + validCharacters[thisDigit];
                aDigit++;
            }
            String strAnswerR = strAnswer.trim();
            System.out.println("ENGLISH--->" + english.get(iTest));
            System.out.println("POLISH--->" + polish.get(iTest));
            System.out.println("TRANSLATED--->" + strAnswerR);
            System.out.println();
        }
        System.out.println("* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * =\n");
    }

    /**
     * A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc
     */
    private static String[] getMinimalCharacterSet() {
        List<String> validChars = new LinkedList<>();
        char[] temp = {' ', '\''};
        for (char c : temp) validChars.add(String.valueOf(c));
        for (char c = 'a'; c <= 'z'; c++) validChars.add(String.valueOf(c));
        String[] out = new String[validChars.size()];
        int i = 0;
        for (String c : validChars) out[i++] = c;
        return out;
    }

}
