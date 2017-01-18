package final_working;

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
    private final int seed;
    private final int batchSize;
    private final int totalBatches;
    private char[] validCharacters;
    private Map<Character, Integer> charToIdxMap;

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
    public TranslatorIterator(int seed, int batchSize, int totalBatches) {

        this.seed = seed;
        this.randomG = new Random(seed);

        this.batchSize = batchSize;
        this.totalBatches = totalBatches;

        this.currentBatch = 0;

        ClassLoader classLoader = Program.class.getClassLoader();
        english = new ArrayList<>();
        polish = new ArrayList<>();
        try {
            FileReader file = new FileReader(classLoader.getResource("translate.txt").getFile());

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
        validCharacters = getMinimalCharacterSet();
        charToIdxMap = new HashMap<>();
        for (int i = 0; i < validCharacters.length; i++) {
            charToIdxMap.put(validCharacters[i], i);
        }
    }

    public char convertIndexToCharacter(int idx) {
        return validCharacters[idx];
    }

    public int convertCharacterToIndex(char c) {
        return charToIdxMap.get(c);
    }

    /**
     * A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc
     */
    public static char[] getMinimalCharacterSet() {
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

    @Override
    public DataSet next(int wordsNumber) {
        polishSelected = new LinkedList<>();
        englishSelected = new LinkedList<>();

        int polishWordsLength = 0;
        int englishWordsLength = 0;
        for (int i = 0; i < wordsNumber; i++) {
            int index = randomG.nextInt(wordsNumber);
            String englishWord = english.get(index);
            englishWordsLength += englishWord.length();
            String polishWord = polish.get(index);
            polishWordsLength += polishWord.length();
            englishSelected.add(englishWord);
            polishSelected.add(polishWord);
        }

        INDArray encoderSeq = Nd4j.zeros(wordsNumber, validCharacters.length * MAX_LINE_LENGTH, MAX_LINE_LENGTH);
        INDArray outputSeq = Nd4j.zeros(wordsNumber, validCharacters.length * MAX_LINE_LENGTH, MAX_LINE_LENGTH);

        for (int i = 0; i < wordsNumber; i++) {
            char[] englishWord = new char[MAX_LINE_LENGTH];
            char[] polishWord = new char[MAX_LINE_LENGTH];
            for (int j = 0; j < englishWord.length; j++) {
                englishWord[j] = ' ';
                polishWord[j] = ' ';
            }
            char[] englishChars = englishSelected.get(i).toCharArray();
            for (int j = 0; j < englishChars.length; j++) {
                englishWord[j] = englishChars[j];
            }
            char[] polishChars = polishSelected.get(i).toCharArray();
            for (int j = 0; j < polishChars.length; j++) {
                polishWord[j] = polishChars[j];
            }

            for (int j = 0; j < englishWord.length; j++) {
                encoderSeq.putScalar(new int[]{i, charToIdxMap.get(englishWord[j]), j}, 1.0);
            }
            for (int j = 0; j < polishWord.length; j++) {
                outputSeq.putScalar(new int[]{i, charToIdxMap.get(polishWord[j]), j}, 1.0);
            }
        }

        //Predict "."
        /* ========================================================================== */
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
        randomG = new Random(seed);
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

    public char[] getValidCharacters() {
        return validCharacters;
    }

    public Map<Character, Integer> getCharToIdxMap() {
        return charToIdxMap;
    }
}
