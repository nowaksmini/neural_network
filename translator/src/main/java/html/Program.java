package html;

import java.io.IOException;
import java.util.List;

public class Program {

    public static void main(String[] args) throws IOException {
        List<String> englishWords = FileUtils.readDataFromFile("normal-words.txt");
        for (String englishWord : englishWords) {
            HtmlUtils.readHtmlFromUri(englishWord);
        }
    }
}
