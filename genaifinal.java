import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Method;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * SRM rules Q&A backend (single file, NO external libraries needed).
 *
 * Uses a Nexus (OpenAI-compatible /chat/completions) endpoint.
 * Required env vars:
 *   NEXUS_API_KEY   your Nexus API key
 *   NEXUS_BASE_URL  e.g. https://your-nexus-host/v1   (the URL that ends right before /chat/completions)
 *   NEXUS_MODEL     model name exactly as Nexus lists it
 *
 * Run:   set NEXUS_API_KEY=your_key      (Windows CMD)
 *        $env:NEXUS_API_KEY="your_key"   (PowerShell)
 *        export NEXUS_API_KEY=your_key   (Mac/Linux)
 *        java NexusQA.java
 *
 * Put .txt files in ./documents  (PDF support is optional, see README notes in the chat).
 */
class NexusQA {

    // ------------------------------ Config ------------------------------
    static final String API_KEY = firstNonBlank(System.getenv("LLM_API_KEY"), System.getenv("NEXUS_API_KEY"));
    static final String MODEL = envOr("LLM_MODEL", envOr("NEXUS_MODEL", "gpt-4.1-nano"));
    static final String BASE_URL = envOr("LLM_BASE_URL", envOr("NEXUS_BASE_URL", "https://nexusapi.navigatelabs.ai")).replaceAll("/+$", "");
    static final String DOCS_DIR = envOr("DOCS_DIR", "documents");
    static final int PORT = Integer.parseInt(envOr("PORT", "8080"));
    static final int TOP_K = 5;
    static final double MIN_SCORE = 0.05;
    static final int MAX_CHUNK_CHARS = 1500;

    static final String NOT_FOUND =
            "The information is not available in the provided documents. "
            + "Please contact the relevant department for assistance.";

    static final String SYSTEM_PROMPT =
            "You are an AI assistant that answers student questions about SRM college rules\n"
            + "using the provided document excerpts (regulations, attendance policy, exam circulars).\n\n"
            + "Rules:\n"
            + "1. Answer ONLY from the provided excerpts. Never fabricate answers or citations.\n"
            + "2. If the answer is not in the excerpts, reply with exactly: \"" + NOT_FOUND + "\"\n"
            + "3. Keep answers student-friendly, professional and clear.\n"
            + "4. If a calculation or checklist is needed (e.g., attendance percentage), show it inline with explanation.\n"
            + "5. If several excerpts support the answer, cite the single most relevant one, using its exact\n"
            + "   document name, clause and page from the excerpt header.\n\n"
            + "Always reply in exactly this format:\n\n"
            + "Answer:\n<your answer in natural language>\n\n"
            + "Citation:\n\uD83D\uDCC4 Source: <Document Name>, Clause <X.Y>, Page <Z>\n";

    static final Pattern CLAUSE = Pattern.compile(
            "^\\s*(?:(?:Clause|Section|Article|Regulation)\\s+)?(\\d+(?:\\.\\d+)+|\\d+[.)])\\s+\\S.*",
            Pattern.CASE_INSENSITIVE);

    static final Set<String> STOP = new HashSet<>(Arrays.asList(
            "a", "an", "the", "is", "are", "am", "was", "were", "be", "to", "of", "in", "on", "for",
            "and", "or", "with", "at", "by", "it", "this", "that", "as", "do", "does", "can", "i",
            "my", "me", "if", "what", "how", "will", "have", "has"));

    static final HttpClient HTTP = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(20)).build();

    static volatile Index index = new Index(new ArrayList<>());

    // ------------------------------ Data ------------------------------
    static final class Chunk {
        final String doc, clause, text;
        final int page;
        Chunk(String doc, String clause, int page, String text) {
            this.doc = doc; this.clause = clause; this.page = page; this.text = text;
        }
    }

    static final class Scored {
        final Chunk chunk;
        final double score;
        Scored(Chunk chunk, double score) { this.chunk = chunk; this.score = score; }
    }

    /** Immutable TF-IDF index snapshot. */
    static final class Index {
        final List<Chunk> chunks;
        final List<Map<String, Double>> vectors = new ArrayList<>();
        final List<Double> norms = new ArrayList<>();
        final Map<String, Double> idf = new HashMap<>();

        Index(List<Chunk> chunks) {
            this.chunks = chunks;
            List<List<String>> tokenized = new ArrayList<>();
            Map<String, Integer> df = new HashMap<>();
            for (Chunk c : chunks) {
                List<String> toks = tokenize(c.text);
                tokenized.add(toks);
                for (String t : new HashSet<>(toks)) df.merge(t, 1, Integer::sum);
            }
            int n = chunks.size();
            for (Map.Entry<String, Integer> e : df.entrySet()) {
                idf.put(e.getKey(), Math.log((1.0 + n) / (1.0 + e.getValue())) + 1.0);
            }
            for (List<String> toks : tokenized) {
                Map<String, Double> v = vectorize(toks);
                vectors.add(v);
                norms.add(norm(v));
            }
        }

        Map<String, Double> vectorize(List<String> toks) {
            Map<String, Double> v = new HashMap<>();
            if (toks.isEmpty()) return v;
            Map<String, Integer> counts = new HashMap<>();
            for (String t : toks) counts.merge(t, 1, Integer::sum);
            for (Map.Entry<String, Integer> e : counts.entrySet()) {
                Double idfVal = idf.get(e.getKey());
                if (idfVal == null) continue;
                double w = ((double) e.getValue() / toks.size()) * idfVal;
                if (w > 0) v.put(e.getKey(), w);
            }
            return v;
        }

        List<Scored> search(String question) {
            List<Scored> result = new ArrayList<>();
            if (chunks.isEmpty()) return result;
            Map<String, Double> qv = vectorize(tokenize(question));
            double qn = norm(qv);
            if (qn == 0) return result;
            for (int i = 0; i < chunks.size(); i++) {
                double cn = norms.get(i);
                if (cn == 0) continue;
                double dot = 0;
                Map<String, Double> cv = vectors.get(i);
                for (Map.Entry<String, Double> e : qv.entrySet()) {
                    Double w = cv.get(e.getKey());
                    if (w != null) dot += w * e.getValue();
                }
                double score = dot / (qn * cn);
                if (score >= MIN_SCORE) result.add(new Scored(chunks.get(i), score));
            }
            result.sort((a, b) -> Double.compare(b.score, a.score));
            return result.size() > TOP_K ? new ArrayList<>(result.subList(0, TOP_K)) : result;
        }
    }

    static double norm(Map<String, Double> v) {
        double s = 0;
        for (Double x : v.values()) s += x * x;
        return Math.sqrt(s);
    }

    static List<String> tokenize(String text) {
        List<String> toks = new ArrayList<>();
        for (String t : text.toLowerCase().split("[^a-z0-9]+")) {
            if (t.isEmpty() || STOP.contains(t)) continue;
            toks.add(t);
        }
        return toks;
    }

    // ------------------------------ Document loading ------------------------------
    static Index buildIndex() {
        List<Chunk> all = new ArrayList<>();
        File dir = new File(DOCS_DIR);
        dir.mkdirs();
        File[] files = dir.listFiles();
        if (files != null) {
            Arrays.sort(files);
            for (File f : files) {
                if (!f.isFile()) continue;
                String lower = f.getName().toLowerCase();
                try {
                    List<String> pages;
                    if (lower.endsWith(".pdf")) {
                        pages = readPdf(f);
                    } else if (lower.endsWith(".txt") || lower.endsWith(".md")) {
                        String content = new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8);
                        pages = Arrays.asList(content.split("\f", -1)); // form feed = page break
                    } else {
                        continue;
                    }
                    String name = f.getName().replaceFirst("\\.[^.]+$", "");
                    all.addAll(chunkDocument(name, pages));
                } catch (Throwable e) {
                    System.out.println("[WARN] Could not read " + f.getName() + ": " + e);
                }
            }
        }
        System.out.println("[INFO] Indexed " + all.size() + " chunks from '" + DOCS_DIR + "'");
        return new Index(all);
    }

    /** Optional PDF support through reflection: works only if the PDFBox 3.x jar is on the classpath. */
    static List<String> readPdf(File f) throws Exception {
        Class<?> loaderCls;
        try {
            loaderCls = Class.forName("org.apache.pdfbox.Loader");
        } catch (ClassNotFoundException e) {
            throw new Exception("PDF support needs pdfbox-app-3.x.jar on the classpath (or convert the PDF to .txt)");
        }
        Class<?> pdDocCls = Class.forName("org.apache.pdfbox.pdmodel.PDDocument");
        Class<?> stripperCls = Class.forName("org.apache.pdfbox.text.PDFTextStripper");
        Object doc = loaderCls.getMethod("loadPDF", File.class).invoke(null, f);
        try {
            Object stripper = stripperCls.getDeclaredConstructor().newInstance();
            Method setStart = stripperCls.getMethod("setStartPage", int.class);
            Method setEnd = stripperCls.getMethod("setEndPage", int.class);
            Method getText = stripperCls.getMethod("getText", pdDocCls);
            int total = (Integer) pdDocCls.getMethod("getNumberOfPages").invoke(doc);
            List<String> pages = new ArrayList<>();
            for (int p = 1; p <= total; p++) {
                setStart.invoke(stripper, p);
                setEnd.invoke(stripper, p);
                pages.add((String) getText.invoke(stripper, doc));
            }
            return pages;
        } finally {
            ((Closeable) doc).close();
        }
    }

    static List<Chunk> chunkDocument(String docName, List<String> pages) {
        List<Chunk> out = new ArrayList<>();
        String currentClause = "N/A";
        for (int p = 0; p < pages.size(); p++) {
            int pageNo = p + 1;
            StringBuilder buf = new StringBuilder();
            String bufClause = currentClause;
            for (String line : pages.get(p).split("\\R")) {
                Matcher m = CLAUSE.matcher(line);
                if (m.matches()) {
                    flush(out, docName, bufClause, pageNo, buf);
                    buf = new StringBuilder();
                    currentClause = m.group(1).replaceAll("[.)]+$", "");
                    bufClause = currentClause;
                }
                buf.append(line).append('\n');
            }
            flush(out, docName, bufClause, pageNo, buf);
        }
        return out;
    }

    static void flush(List<Chunk> out, String doc, String clause, int page, StringBuilder buf) {
        String text = buf.toString().trim();
        if (text.isEmpty()) return;
        for (int i = 0; i < text.length(); i += MAX_CHUNK_CHARS) {
            out.add(new Chunk(doc, clause, page, text.substring(i, Math.min(text.length(), i + MAX_CHUNK_CHARS))));
        }
    }

    // ------------------------------ Nexus (OpenAI-compatible chat completions) ------------------------------
    static String callLlm(String systemPrompt, String userPrompt) throws Exception {
        String body = "{\"model\":" + q(MODEL) + ",\"temperature\":0.1,\"messages\":["
                + "{\"role\":\"system\",\"content\":" + q(systemPrompt) + "},"
                + "{\"role\":\"user\",\"content\":" + q(userPrompt) + "}]}";
        String url = BASE_URL + "/chat/completions";
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + API_KEY)
                .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .build();
        HttpResponse<String> resp = HTTP.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (resp.statusCode() != 200) {
            throw new Exception("Nexus returned HTTP " + resp.statusCode() + ": " + resp.body());
        }
        Object root = Json.parse(resp.body());
        if (root instanceof Map) {
            Object choices = ((Map<?, ?>) root).get("choices");
            if (choices instanceof List && !((List<?>) choices).isEmpty()) {
                Object choice = ((List<?>) choices).get(0);
                if (choice instanceof Map) {
                    Object msg = ((Map<?, ?>) choice).get("message");
                    if (msg instanceof Map) {
                        Object content = ((Map<?, ?>) msg).get("content");
                        if (content instanceof String) return ((String) content).trim();
                    }
                }
            }
        }
        return "";
    }

    // ------------------------------ HTTP handlers ------------------------------
    interface Handler { void run(HttpExchange ex) throws Exception; }

    static HttpHandler safe(Handler h) {
        return ex -> {
            try {
                if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                    send(ex, 204, "");
                    return;
                }
                h.run(ex);
            } catch (Throwable e) {
                try {
                    send(ex, 500, "{\"error\":" + q("Server error: " + e) + "}");
                } catch (Throwable ignored) { }
            }
        };
    }

    static void send(HttpExchange ex, int status, String json) throws IOException {
        byte[] b = json.getBytes(StandardCharsets.UTF_8);
        Headers h = ex.getResponseHeaders();
        h.set("Content-Type", "application/json; charset=utf-8");
        h.set("Access-Control-Allow-Origin", "*");
        h.set("Access-Control-Allow-Headers", "Content-Type");
        h.set("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
        if (status == 204) {
            ex.sendResponseHeaders(204, -1);
            ex.close();
            return;
        }
        ex.sendResponseHeaders(status, b.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(b);
        }
    }

    static void health(HttpExchange ex) throws Exception {
        send(ex, 200, "{\"status\":\"ok\",\"chunksIndexed\":" + index.chunks.size()
                + ",\"model\":" + q(MODEL) + ",\"baseUrl\":" + q(BASE_URL)
                + ",\"nexusConfigured\":" + (API_KEY != null && !BASE_URL.isEmpty() && !MODEL.isEmpty()) + "}");
    }

    static void reload(HttpExchange ex) throws Exception {
        index = buildIndex();
        send(ex, 200, "{\"chunksIndexed\":" + index.chunks.size() + "}");
    }

    static void ask(HttpExchange ex) throws Exception {
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            send(ex, 405, "{\"error\":\"Use POST\"}");
            return;
        }
        String raw = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        String question = null;
        try {
            Object parsed = Json.parse(raw);
            if (parsed instanceof Map) {
                Object qv = ((Map<?, ?>) parsed).get("question");
                if (qv instanceof String) question = ((String) qv).trim();
            }
        } catch (Exception e) {
            send(ex, 400, "{\"error\":\"Invalid JSON body. Send {\\\"question\\\": \\\"...\\\"}\"}");
            return;
        }
        if (question == null || question.isEmpty()) {
            send(ex, 400, "{\"error\":\"Question cannot be empty.\"}");
            return;
        }

        List<Scored> hits = index.search(question);
        if (hits.isEmpty()) {
            send(ex, 200, "{\"answer\":" + q(NOT_FOUND) + ",\"sources\":[]}");
            return;
        }
        if (API_KEY == null || BASE_URL.isEmpty() || MODEL.isEmpty()) {
            send(ex, 500, "{\"error\":\"Set NEXUS_API_KEY, NEXUS_BASE_URL and NEXUS_MODEL on the server.\"}");
            return;
        }

        StringBuilder context = new StringBuilder();
        for (Scored s : hits) {
            Chunk c = s.chunk;
            context.append("Document: ").append(c.doc).append('\n')
                    .append("Clause: ").append(c.clause).append('\n')
                    .append("Page: ").append(c.page).append('\n')
                    .append("Text: ").append(c.text).append("\n---\n");
        }
        String prompt = "Document excerpts:\n" + context + "\nStudent question: " + question;

        String answer;
        try {
            answer = callLlm(SYSTEM_PROMPT, prompt);
        } catch (Exception e) {
            send(ex, 502, "{\"error\":" + q("Nexus API error: " + e.getMessage()) + "}");
            return;
        }
        if (answer.isEmpty() || answer.toLowerCase().contains(NOT_FOUND.toLowerCase())) {
            send(ex, 200, "{\"answer\":" + q(NOT_FOUND) + ",\"sources\":[]}");
            return;
        }

        StringBuilder src = new StringBuilder("[");
        for (int i = 0; i < hits.size(); i++) {
            Scored s = hits.get(i);
            if (i > 0) src.append(',');
            src.append("{\"docName\":").append(q(s.chunk.doc))
                    .append(",\"clause\":").append(q(s.chunk.clause))
                    .append(",\"page\":").append(s.chunk.page)
                    .append(",\"score\":").append(Math.round(s.score * 1000.0) / 1000.0).append('}');
        }
        src.append(']');
        send(ex, 200, "{\"answer\":" + q(answer) + ",\"sources\":" + src + "}");
    }

    // ------------------------------ Main ------------------------------
    public static void main(String[] args) throws Exception {
        index = buildIndex();
        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/health", safe(NexusQA::health));
        server.createContext("/reload", safe(NexusQA::reload));
        server.createContext("/ask", safe(NexusQA::ask));
        server.createContext("/", ex -> {
            try {
                byte[] b = java.nio.file.Files.readAllBytes(java.nio.file.Paths.get("index.html"));
                ex.getResponseHeaders().add("Content-Type", "text/html; charset=utf-8");
                ex.sendResponseHeaders(200, b.length);
                ex.getResponseBody().write(b);
            } catch (Exception e) { ex.sendResponseHeaders(404, -1); }
            ex.close();
        });
        server.setExecutor(Executors.newFixedThreadPool(8));
        server.start();
        System.out.println("[INFO] Server running on http://localhost:" + PORT);
        if (API_KEY == null) System.out.println("[WARN] NEXUS_API_KEY is not set.");
        if (BASE_URL.isEmpty()) System.out.println("[WARN] NEXUS_BASE_URL is not set.");
        if (MODEL.isEmpty()) System.out.println("[WARN] NEXUS_MODEL is not set.");
    }

    // ------------------------------ Small utilities ------------------------------
    static String envOr(String name, String def) {
        String v = System.getenv(name);
        return (v == null || v.trim().isEmpty()) ? def : v.trim();
    }

    static String firstNonBlank(String a, String b) {
        if (a != null && !a.trim().isEmpty()) return a.trim();
        if (b != null && !b.trim().isEmpty()) return b.trim();
        return null;
    }

    /** JSON string quoting. */
    static String q(String s) {
        StringBuilder b = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default:
                    if (c < 0x20) b.append(String.format("\\u%04x", (int) c));
                    else b.append(c);
            }
        }
        return b.append('"').toString();
    }

    /** Minimal JSON parser (objects -> Map, arrays -> List, strings, numbers, booleans, null). */
    static final class Json {
        private final String s;
        private int i = 0;

        private Json(String s) { this.s = s; }

        static Object parse(String s) {
            Json j = new Json(s);
            j.ws();
            Object v = j.value();
            j.ws();
            if (j.i != s.length()) throw new IllegalArgumentException("Unexpected trailing data");
            return v;
        }

        private void ws() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
        }

        private Object value() {
            if (i >= s.length()) throw new IllegalArgumentException("Unexpected end of JSON");
            char c = s.charAt(i);
            if (c == '{') return obj();
            if (c == '[') return arr();
            if (c == '"') return str();
            if (s.startsWith("true", i)) { i += 4; return Boolean.TRUE; }
            if (s.startsWith("false", i)) { i += 5; return Boolean.FALSE; }
            if (s.startsWith("null", i)) { i += 4; return null; }
            return num();
        }

        private Map<String, Object> obj() {
            Map<String, Object> m = new LinkedHashMap<>();
            i++; // {
            ws();
            if (s.charAt(i) == '}') { i++; return m; }
            while (true) {
                ws();
                String k = str();
                ws();
                if (s.charAt(i++) != ':') throw new IllegalArgumentException("Expected ':'");
                ws();
                m.put(k, value());
                ws();
                char c = s.charAt(i++);
                if (c == ',') continue;
                if (c == '}') return m;
                throw new IllegalArgumentException("Expected ',' or '}'");
            }
        }

        private List<Object> arr() {
            List<Object> l = new ArrayList<>();
            i++; // [
            ws();
            if (s.charAt(i) == ']') { i++; return l; }
            while (true) {
                ws();
                l.add(value());
                ws();
                char c = s.charAt(i++);
                if (c == ',') continue;
                if (c == ']') return l;
                throw new IllegalArgumentException("Expected ',' or ']'");
            }
        }

        private String str() {
            if (s.charAt(i) != '"') throw new IllegalArgumentException("Expected string");
            i++;
            StringBuilder b = new StringBuilder();
            while (true) {
                char c = s.charAt(i++);
                if (c == '"') return b.toString();
                if (c == '\\') {
                    char e = s.charAt(i++);
                    switch (e) {
                        case '"': b.append('"'); break;
                        case '\\': b.append('\\'); break;
                        case '/': b.append('/'); break;
                        case 'b': b.append('\b'); break;
                        case 'f': b.append('\f'); break;
                        case 'n': b.append('\n'); break;
                        case 'r': b.append('\r'); break;
                        case 't': b.append('\t'); break;
                        case 'u':
                            b.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                            i += 4;
                            break;
                        default: throw new IllegalArgumentException("Bad escape");
                    }
                } else {
                    b.append(c);
                }
            }
        }

        private Double num() {
            int start = i;
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
            if (start == i) throw new IllegalArgumentException("Unexpected character at " + i);
            return Double.valueOf(s.substring(start, i));
        }
    }
}