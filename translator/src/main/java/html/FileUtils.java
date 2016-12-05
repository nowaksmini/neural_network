package html;

import java.io.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

public class FileUtils {

    public static void generateEnglishDictionary() throws IOException {
        List<String> normalWords = new LinkedList<>();
        List<String> strangeWords = new LinkedList<>();
        FileUtils.readDataFromFile("words.txt", normalWords, strangeWords);
        FileUtils.writeDataToFile("strange-words.txt", strangeWords.toArray());
        FileUtils.writeDataToFile("normal-words.txt", normalWords.toArray());
    }

    private static void readDataFromFile(String fileName, List<String> normalWords, List<String> strangeWords) {

        //Get file from resources folder
        ClassLoader classLoader = Program.class.getClassLoader();
        File file = new File(classLoader.getResource(fileName).getFile());
        final String[] strangeSigns = {"'", "\"", ";", ".", "~", "/", "@", "$", "%", "*", "&",
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

        try (Scanner scanner = new Scanner(file)) {

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                boolean normal = true;
                for (String strangeSign : strangeSigns) {
                    if (line.contains(strangeSign)) {
                        normal = false;
                        break;
                    }
                }
                if (normal) {
                    normalWords.add(line);
                } else {
                    strangeWords.add(line);
                }

            }

            scanner.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static List<String> readDataFromFile(String fileName) {
        List<String> output = new LinkedList<>();
        ClassLoader classLoader = Program.class.getClassLoader();
        File file = new File(classLoader.getResource(fileName).getFile());

        try (Scanner scanner = new Scanner(file)) {

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                output.add(line);
            }
            scanner.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
        return output;
    }

    /**
     * saves data in /target catalog in classes
     */
    public static void writeDataToFile(String fileName, Object[] data) throws IOException {
        ClassLoader classLoader = Program.class.getClassLoader();
        FileWriter file = new FileWriter(classLoader.getResource(fileName).getFile(), true);

        BufferedWriter bw = new BufferedWriter(file);

        for (Object aData : data) {
            bw.write(aData.toString());
            bw.newLine();
        }
        bw.close();
    }
}
