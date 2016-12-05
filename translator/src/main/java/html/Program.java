package html;

import java.io.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

public class Program {

    public static void main(String[] args) throws IOException {
        List<String> normalWords = new LinkedList<>();
        List<String> strangeWords = new LinkedList<>();
        readDataFromFile("words.txt", normalWords, strangeWords);
        writeDataToFile("strange-words.txt", strangeWords.toArray());
        writeDataToFile("normal-words.txt", normalWords.toArray());
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

    /**
     * saves data in /out catalog
     */
    private static void writeDataToFile(String fileName, Object[] data) throws IOException {
        ClassLoader classLoader = Program.class.getClassLoader();
        File file = new File(classLoader.getResource(fileName).getFile());
        FileOutputStream fos = new FileOutputStream(file);

        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));

        for (Object aData : data) {
            bw.write(aData.toString());
            bw.newLine();
        }
        bw.close();
    }

}
