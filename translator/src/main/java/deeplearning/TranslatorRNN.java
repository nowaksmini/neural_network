package deeplearning;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.*;


public class TranslatorRNN {
    private static final String NETWORK_PATH = "network.txt";

    public static void main(String[] args) throws Exception {
        trainNetwork();
        // TODO fix test network
//        List<String> englishTest = new LinkedList<>();
//        englishTest.add("tea");
//        englishTest.add("tea");
//        testNetwork(englishTest);
    }

    private static void trainNetwork() {
        int miniBatchSize = 4; // Number of file lines used when training
        int numHiddenNodes = 128;
        String[] validCharacters = getMinimalCharacterSet();
        int FEATURE_VEC_SIZE = validCharacters.length * TranslatorIterator.MAX_LINE_LENGTH;
        int seed = 123467;
        TranslatorIterator shakespeareIterator = new TranslatorIterator(seed, miniBatchSize, 3, validCharacters);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                .learningRate(0.5)
                .updater(Updater.RMSPROP)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .seed(seed)
                .graphBuilder()
                .addInputs("additionIn")
                .setInputTypes(InputType.recurrent(FEATURE_VEC_SIZE))
                .addLayer("encoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE).nOut(numHiddenNodes).activation("softsign").build(), "additionIn")
                .addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
                .addLayer("output", new RnnOutputLayer.Builder().nIn(numHiddenNodes).nOut(FEATURE_VEC_SIZE).activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT).build(), "encoder")
                .setOutputs("output")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        //Train model:
        int iEpoch = 0;
        while (iEpoch < 2) {
            while (shakespeareIterator.hasNext()) {
                DataSet ds = shakespeareIterator.next();
                MultiDataSet multiDataSet = new MultiDataSet(ds.getFeatures(), ds.getLabels());
                net.fit(multiDataSet);
                INDArray[] output = net.output(ds.getFeatures());
                List<String> englishSelected = shakespeareIterator.getEnglishSelected();
                List<String> polishSelected = shakespeareIterator.getPolishSelected();
                INDArray predictions = output[0];
                INDArray answers = Nd4j.argMax(predictions, 1);
                checkAnswers(englishSelected, polishSelected, answers, shakespeareIterator.getValidCharacters());
            }
            iEpoch++;
            shakespeareIterator.reset();
            System.out.println("\n* = * = * = * = * = * = * = * = * = ** EPOCH " + iEpoch + " COMPLETE ** = * = * = * = * = * = * = * = * = * = * = * = * = * =\n");
        }

        saveNeuralNetwork(net);

        System.out.println("\n\nNetwork saved");
    }

    private static void testNetwork(List<String> englishInput) {
        ComputationGraph net;
        try {
            FileInputStream file = new FileInputStream(NETWORK_PATH);
            ObjectInputStream objectOutputStream = new ObjectInputStream(file);
            net = (ComputationGraph) objectOutputStream.readObject();
            objectOutputStream.close();
            file.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return;
        }
        String[] validCharacters = getMinimalCharacterSet();
        Map<String, Integer> charToIdxMap = new HashMap<>();
        for (int i = 0; i < validCharacters.length; i++) {
            charToIdxMap.put(validCharacters[i], i);
        }
        int wordsNumber = 1;
        INDArray encoderSeq = Nd4j.zeros(wordsNumber, validCharacters.length * TranslatorIterator.MAX_LINE_LENGTH,
                TranslatorIterator.MAX_LINE_LENGTH);

        for (int i = 0; i < wordsNumber; i++) {
            String englishWord = englishInput.get(0);
            for (int j = englishWord.length(); j < TranslatorIterator.MAX_LINE_LENGTH; j++) {
                englishWord += ' ';
            }
            String[] englishSplit = englishWord.split("");
            for (int j = 0; j < englishSplit.length; j++) {
                Integer integer = charToIdxMap.get(englishSplit[j]);
                if (integer == null) {
                    integer = charToIdxMap.get(" ");
                }
                encoderSeq.putScalar(new int[]{i, integer, j}, 1.0);
            }
        }
        INDArray[] output = net.output(encoderSeq);
        checkAnswers(englishInput, englishInput, output[0], validCharacters);
    }

    private static void saveNeuralNetwork(ComputationGraph computationGraph) {
        try {
            FileOutputStream file = new FileOutputStream(NETWORK_PATH);
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(file);
            objectOutputStream.writeObject(computationGraph);
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
        char[] temp = {' ', '!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', '\n', '\t'};
        for (char c : temp) validChars.add(String.valueOf(c));
        for (char c = 'a'; c <= 'z'; c++) validChars.add(String.valueOf(c));
        for (char c = 'A'; c <= 'Z'; c++) validChars.add(String.valueOf(c));
        for (char c = '0'; c <= '9'; c++) validChars.add(String.valueOf(c));
        //polish = {"¹", "æ", "ê", "³", "ñ", "ó", "œ", "Ÿ", "¿"};
        char[] polish = {'\u0105', '\u0107', '\u0119', '\u0142', '\u0144', '\u00F3', '\u015B', '\u017A', '\u017C',
                '\u0104', '\u0106', '\u0118', '\u0141', '\u0143', '\u00D3', '\u015A', '\u0179', '\u017B'};
        for (char s : polish) {
            validChars.add(String.valueOf(s));
        }
        String[] out = new String[validChars.size()];
        int i = 0;
        for (String c : validChars) out[i++] = c;
        return out;
    }

}
