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

import java.util.LinkedList;
import java.util.List;


public class TranslatorRNN {

    public static void main(String[] args) throws Exception {

        int miniBatchSize = 3; // Number of file lines used when  training
        int numHiddenNodes = 128;
        char[] validCharacters = getMinimalCharacterSet();
        int FEATURE_VEC_SIZE = validCharacters.length * TranslatorIterator.MAX_LINE_LENGTH; // 30 * 77
        TranslatorIterator shakespeareIterator = new TranslatorIterator(12345, miniBatchSize, 3, validCharacters);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                .learningRate(0.5)
                .updater(Updater.RMSPROP)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .seed(12345)
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
        while (iEpoch < 5) {
            while (shakespeareIterator.hasNext()) {
                DataSet ds = shakespeareIterator.next();
                MultiDataSet multiDataSet = new MultiDataSet(ds.getFeatures(), ds.getLabels());
                net.fit(multiDataSet);
                org.nd4j.linalg.api.ndarray.INDArray[] output = net.output(ds.getFeatures());
                List<String> englishSelected = shakespeareIterator.getEnglishSelected();
                List<String> polishSelected = shakespeareIterator.getPolishSelected();
                INDArray predictions = output[0];
                INDArray answers = Nd4j.argMax(predictions, 1);
                checkAnswers(englishSelected, polishSelected, answers, shakespeareIterator.getValidCharacters());
            }
            iEpoch++;
            shakespeareIterator.reset();
        }
        System.out.println("\n* = * = * = * = * = * = * = * = * = ** EPOCH " + iEpoch + " COMPLETE ** = * = * = * = * = * = * = * = * = * = * = * = * = * =");

        System.out.println("\n\nExample complete");
    }

    private static void checkAnswers(List<String> english, List<String> polish, INDArray answers,
                                     char[] validCharacters) {
        int nTests = answers.size(0);
        for (int iTest = 0; iTest < nTests; iTest++) {
            int aDigit = 0;
            String strAnswer = "";
            while (aDigit < TranslatorIterator.MAX_LINE_LENGTH) {
                int thisDigit = answers.getInt(iTest, aDigit);
                strAnswer = strAnswer + validCharacters[thisDigit];
                aDigit++;
            }
            String strAnswerR = strAnswer.trim();
            System.out.println("ENGLISH--->" + english.get(iTest));
            System.out.println("POLISH--->" + polish.get(iTest));
            System.out.println("TRANSLATED--->" + strAnswerR);
            System.out.println();
        }
        System.out.println("*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*");
    }

    /**
     * A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc
     */
    private static char[] getMinimalCharacterSet() {
        List<Character> validChars = new LinkedList<>();
        for (char c = 'a'; c <= 'z'; c++) validChars.add(c);
        for (char c = 'A'; c <= 'Z'; c++) validChars.add(c);
        for (char c = '0'; c <= '9'; c++) validChars.add(c);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for (char c : temp) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i = 0;
        for (Character c : validChars) out[i++] = c;
        return out;
    }

}
