package html;

import java.io.IOException;
import java.util.List;

public class Program {

    public static void main(String[] args) {

    }

    private static void fillTestData() throws IOException {
        List<String> englishWords = FileUtils.readDataFromFile("simple-words.txt");
        for (String englishWord : englishWords) {
            HtmlUtils.readHtmlFromUri(englishWord);
        }
    }
}
