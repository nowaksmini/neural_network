package html;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.IOException;
import java.net.URL;

public class HtmlUtils {

    public static void readHtmlFromUri(String word) throws IOException {
        URL dictionaryUrl = new URL("http://pl.bab.la/slownik/angielski-polski/" + word);
        Document doc = Jsoup.parse(dictionaryUrl, 10000);
        Elements resultBlocks = doc.select("div.result-block");
        for (Element resultBlock : resultBlocks) {
            Elements links = resultBlock.getElementsByClass("result-wrapper");
            for (Element link : links) {
                //System.out.println("--------------------------------------------");
                Elements elementsByClass = link.getElementsByClass("result-link");
                if (elementsByClass != null) {
                    if (elementsByClass.size() == 2) {
                        String translation = elementsByClass.get(1).text();
                        String[] split = translation.split(",");
                        for (String s : split) {
                            String dictionaryRecord = elementsByClass.get(0).text() + " ---> " + s.trim();
                            //System.out.println("SŁOWNIK - " + dictionaryRecord);
                            FileUtils.writeDataToFile("dictionary.txt", new Object[]{dictionaryRecord});
                        }
                    } else {
                        Elements additionalPhraseENG = link.getElementsByClass("result-left");
                        Elements additionalPhrasePL = link.getElementsByClass("result-right");
                        if ((additionalPhraseENG == null || additionalPhraseENG.size() == 0) ||
                                (additionalPhrasePL == null || additionalPhrasePL.size() == 0)) {
                            // SYNONIMY
                        } else {
                            Elements volumeSymbol = additionalPhraseENG.get(0).getElementsByClass("warn-cs");
                            if (volumeSymbol == null || volumeSymbol.isEmpty()) {
                                //System.out.println("DODATKOWE TŁUMACZENIA");
                                String sentenceTranslation = additionalPhraseENG.text() + " ---> " + additionalPhrasePL.text();
                                //System.out.println(sentenceTranslation);
                                FileUtils.writeDataToFile("sentences.txt", new Object[]{sentenceTranslation});
                            } else {
                                //System.out.println("DODATKOWA FRAZA");
                                String[] split = additionalPhrasePL.text().split(",");
                                for (String s : split) {
                                    String phraseTranslation = additionalPhraseENG.text() + " ---> " + s.trim();
                                    //(phraseTranslation);
                                    FileUtils.writeDataToFile("phrases.txt", new Object[]{phraseTranslation});
                                }
                            }
                        }
                    }
                } else {
                    System.out.println("ERROR - for word " + word + " no result link");
                }
                Elements examplePL = link.getElementsByClass("span6 result-right row-fluid babCSColor");
                Elements exampleEN = link.getElementsByClass("span6 result-left");
                if (examplePL != null && exampleEN != null && !examplePL.isEmpty() && !exampleEN.isEmpty() && exampleEN.size() > 1
                        && examplePL.size() == (exampleEN.size() - 1)) {
                    //System.out.println("TŁUMACZENIA");
                    for (int i = 0; i < examplePL.size(); i++) {
                        String sentenceTranslation = exampleEN.get(i + 1).text() + " ---> " + examplePL.get(i).text();
                        //System.out.println(sentenceTranslation);
                        FileUtils.writeDataToFile("sentences.txt", new Object[]{sentenceTranslation});
                    }
                }
            }
        }
        System.out.println("FINISHED - " + word);
    }
}
