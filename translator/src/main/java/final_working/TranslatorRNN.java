package final_working;

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

import java.util.List;


/**
 * Created by susaneraly on 3/27/16.
 */
public class TranslatorRNN {

    public static void main(String[] args) throws Exception {

        int lstmLayerSize = 77;                    //Number of units in each GravesLSTM layer
        int miniBatchSize = 3;                        //Size of mini batch to use when  training
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 1;                            //Total number of training epochs
        TranslatorIterator shakespeareIterator = new TranslatorIterator(12345, miniBatchSize, 3);
        int nOut = shakespeareIterator.totalOutcomes();
        int numHiddenNodes = 128;
        int FEATURE_VEC_SIZE = 2310; // 30 * 77

        //Set up network configuration:
        //MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
//                .learningRate(0.1)
//                .rmsDecay(0.95)
//                .seed(12345)
//                .regularization(true)
//                .l2(0.001)
//                .weightInit(WeightInit.XAVIER)
//                .updater(Updater.RMSPROP)
//                .list()
//                .layer(0, new GravesLSTM.Builder().nIn(shakespeareIterator.inputColumns()).nOut(lstmLayerSize)
//                        .activation("tanh").build())
//                .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
//                        .activation("tanh").build())
//                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
//                        .nIn(lstmLayerSize).nOut(nOut).build())
//                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
//                .pretrain(false).backprop(true)
//                .build();
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                .learningRate(0.5)
                .updater(Updater.RMSPROP)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .seed(12345)
                .graphBuilder()
                .addInputs("additionIn")
                .setInputTypes(InputType.recurrent(FEATURE_VEC_SIZE))
                .addLayer("encoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE).nOut(numHiddenNodes).activation("softsign").build(),"additionIn")
                .addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
                //.addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
                //.addLayer("decoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE+numHiddenNodes).nOut(numHiddenNodes).activation("softsign").build(), "sumOut","duplicateTimeStep")
                .addLayer("output", new RnnOutputLayer.Builder().nIn(numHiddenNodes).nOut(FEATURE_VEC_SIZE).activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT).build(), "encoder")
                .setOutputs("output")
                .pretrain(false).backprop(true)
                .build();

//        MultiLayerNetwork net = new MultiLayerNetwork(conf);
//        net.init();
//        net.setListeners(new ScoreIterationListener(1));

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        //Train model:
        int iEpoch = 0;
        int testSize = 200;
        while (iEpoch < 5) {
            while (shakespeareIterator.hasNext()) {
                DataSet ds = shakespeareIterator.next();
                MultiDataSet multiDataSet = new MultiDataSet(ds.getFeatures(), ds.getLabels());
                net.fit(multiDataSet);
                org.nd4j.linalg.api.ndarray.INDArray[] output = net.output(ds.getFeatures());
                List<String> englishSelected = shakespeareIterator.getEnglishSelected();
                List<String> polishSelected = shakespeareIterator.getPolishSelected();
                INDArray predictions = output[0];
                INDArray answers = Nd4j.argMax(predictions,1);
                checkAnswers(englishSelected, polishSelected, answers,shakespeareIterator.getValidCharacters());
            }
            iEpoch++;
            shakespeareIterator.reset();
        }
        System.out.println("\n* = * = * = * = * = * = * = * = * = ** EPOCH " + iEpoch + " COMPLETE ** = * = * = * = * = * = * = * = * = * = * = * = * = * =");

//        //Print the  number of parameters in the network (and for each layer)
//        Layer[] layers = net.getLayers();
//        int totalNumParams = 0;
//        for (int i = 0; i < layers.length; i++) {
//            int nParams = layers[i].numParams();
//            System.out.println("Number of parameters in layer " + i + ": " + nParams);
//            totalNumParams += nParams;
//        }
//        System.out.println("Total number of network parameters: " + totalNumParams);
//
//        //Do training, and then generate and print samples from network
//        int miniBatchNumber = 0;
//        for (int i = 0; i < numEpochs; i++) {
//            while (shakespeareIterator.hasNext()) {
//                DataSet ds = shakespeareIterator.next();
//                System.out.println("---------BEFORE-----------");
//                net.fit(ds);
//                INDArray output = net.output(ds.getFeatures());
//                List<String> englishSelected = shakespeareIterator.getEnglishSelected();
//                List<String> polishSelected = shakespeareIterator.getPolishSelected();
//                checkAnswers(englishSelected, polishSelected, output);
//                System.out.println("---------AFTER-----------");
//
//                System.out.println("--------------------");
//                System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize);
//            }
//
//            shakespeareIterator.reset();
//        }

        System.out.println("\n\nExample complete");
    }

    private static void checkAnswers(List<String> english, List<String> polish, INDArray answers,
                                     char[] validCharacters) {
        int nTests = answers.size(0);
        for (int iTest=0; iTest < nTests; iTest++) {
            int aDigit = 0;
            String strAnswer = "";
            while (aDigit < 30) {
                int thisDigit = answers.getInt(iTest,aDigit);
                //System.out.println(thisDigit);
                strAnswer = strAnswer + validCharacters[thisDigit];
                aDigit++;
            }
            String strAnswerR = strAnswer.trim();
            System.out.println("ENGLISH--->" + english.get(iTest));
            System.out.println("POLISH--->" + polish.get(iTest));
            System.out.println(strAnswerR);
        }
        System.out.println("*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*");
    }

}
