package deeplearning;

import html.Program;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class TranslatorIterator implements DataSetIterator {

    private Random randomG;
    private int currentBatch; //number of actual training
    private final int batchSize;
    private final int totalBatches;
    private String[] validCharacters;
    private Map<String, Integer> charToIdxMap;

    /**
     * List of english sequences to translate
     */
    private final List<String> english;
    /**
     * List of polish sequences translating @see english, in good order
     */
    private final List<String> polish;

    private List<String> englishSelected = new LinkedList<>();
    private List<String> polishSelected = new LinkedList<>();

    public static final int MAX_LINE_LENGTH = 30;

    /**
     * creates TranslatorIterator
     *
     * @param seed         for random (we want to reproduce test)
     * @param batchSize    number of words in one learning test
     * @param totalBatches number of all tests e.g dictionary has 200 words, batchSize=10, totalBatches=20
     */
    public TranslatorIterator(int seed, int batchSize, int totalBatches, String[] validCharacters) {

        this.randomG = new Random(seed);

        this.batchSize = batchSize;
        this.totalBatches = totalBatches;

        this.currentBatch = 0;

        ClassLoader classLoader = Program.class.getClassLoader();
        english = new ArrayList<>();
        polish = new ArrayList<>();
        try {
            FileReader file = new FileReader(classLoader.getResource(TranslatorProgram.TRAINING_SET_PATH).getFile());

            BufferedReader bw = new BufferedReader(file);
            String line;
            while ((line = bw.readLine()) != null) {
                String[] split = line.split("--->");
                String englishWord = split[0].trim();
                String polishWord = split[1].trim();
                english.add(englishWord);
                polish.add(polishWord);
                System.out.println("DICTIONARY :" + englishWord + "--->" + polishWord);
            }
            bw.close();
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.validCharacters = validCharacters;
        charToIdxMap = new HashMap<>();
        for (int i = 0; i < validCharacters.length; i++) {
            charToIdxMap.put(validCharacters[i], i);
        }
    }

    public TranslatorIterator(int seed, int batchSize, int totalBatches, String[] validCharacters,
                              List<String> englishWords, List<String> polishWords) {

        this.randomG = new Random(seed);
        this.batchSize = batchSize;
        this.totalBatches = totalBatches;
        this.currentBatch = 0;
        english = englishWords;
        polish = polishWords;
        this.validCharacters = validCharacters;
        charToIdxMap = new HashMap<>();
        for (int i = 0; i < validCharacters.length; i++) {
            charToIdxMap.put(validCharacters[i], i);
        }
    }

    @Override
    public DataSet next(int wordsNumber) {
        polishSelected = new LinkedList<>();
        englishSelected = new LinkedList<>();

        for (int i = 0; i < wordsNumber; i++) {
            int index = randomG.nextInt(english.size());
            String englishWord = english.get(index);
            String polishWord = polish.get(index);
            englishSelected.add(englishWord);
            polishSelected.add(polishWord);
        }

        INDArray encoderSeq = Nd4j.zeros(wordsNumber, validCharacters.length * MAX_LINE_LENGTH, MAX_LINE_LENGTH);
        INDArray outputSeq = Nd4j.zeros(wordsNumber, validCharacters.length * MAX_LINE_LENGTH, MAX_LINE_LENGTH);

        for (int i = 0; i < wordsNumber; i++) {
            String englishWord = englishSelected.get(i);
            String polishWord = polishSelected.get(i);
            for (int j = englishWord.length(); j < MAX_LINE_LENGTH; j++) {
                englishWord += ' ';
            }
            for (int j = polishWord.length(); j < MAX_LINE_LENGTH; j++) {
                polishWord += ' ';
            }
            String[] polishSplit = polishWord.split("");
            String[] englishSplit = englishWord.split("");
            for (int j = 0; j < englishSplit.length; j++) {
                Integer integer = charToIdxMap.get(englishSplit[j]);
                if (integer == null) {
                    integer = charToIdxMap.get(" ");
                }
                encoderSeq.putScalar(new int[]{i, integer, j}, 1.0);
            }
            for (int j = 0; j < polishSplit.length; j++) {
                Integer integer = charToIdxMap.get(polishSplit[j]);
                if (integer == null) {
                    integer = charToIdxMap.get(" ");
                }
                outputSeq.putScalar(new int[]{i, integer, j}, 1.0);
            }
        }

        currentBatch++;
        return new DataSet(encoderSeq, outputSeq);
    }

    @Override
    public int totalExamples() {
        return english.size();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void reset() {
        currentBatch = 0;
        //randomG = new Random(seed);
    }

    @Override
    public boolean hasNext() {
        return currentBatch < totalBatches;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public int inputColumns() {
        return validCharacters.length * MAX_LINE_LENGTH;
    }

    @Override
    public int totalOutcomes() {
        return validCharacters.length * MAX_LINE_LENGTH;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }


    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return english.size() - currentBatch * batchSize;
    }


    @Override
    public int numExamples() {
        return english.size();
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }


    public List<String> getEnglishSelected() {
        return englishSelected;
    }

    public List<String> getPolishSelected() {
        return polishSelected;
    }


    public String[] getValidCharacters() {
        return validCharacters;
    }

    public Map<String, Integer> getCharToIdxMap() {
        return charToIdxMap;
    }
}
