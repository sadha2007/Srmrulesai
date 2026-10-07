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
 * TWO MODES:
 *   FREE MODE (default, no key, no internet needed):
 *       Returns the best matching rule text with its citation.
 *   AI MODE (optional, free Gemini key from aistudio.google.com):
 *       Returns a clean, student-friendly answer with a citation.
 *
 * Run:   java GeminiQA.java                         (FREE MODE)
 *
 * Optional AI mode - set the key in the SAME terminal first:
 *        set GEMINI_API_KEY=your_key      (Windows CMD)
 *        $env:GEMINI_API_KEY="your_key"   (PowerShell)
 *        export GEMINI_API_KEY=your_key   (Mac/Linux)
 *        java GeminiQA.java
 *
 * Put .txt files in ./documents  (PDF support is optional, needs pdfbox jar).
 */
class GeminiQA {

    // ------------------------------ Config ------------------------------
    static final String API_KEY = firstNonBlank(System.getenv("GEMINI_API_KEY"), "AQ.Ab8RN6KyoBN1O_akrtDUxI-wu_V_-RL-H-RNntZ-UBfuObFxig");
    static final String MODEL = envOr("GEMINI_MODEL", "gemini-3.5-flash-lite");
    static final String BASE_URL = envOr("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com");
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

    // ------------------------------ Gemini (optional AI mode) ------------------------------
    static String callGemini(String systemPrompt, String userPrompt) throws Exception {
        String body = "{\"systemInstruction\":{\"parts\":[{\"text\":" + q(systemPrompt) + "}]},"
                + "\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":" + q(userPrompt) + "}]}],"
                + "\"generationConfig\":{\"temperature\":0.1}}";
        String url = BASE_URL + "/v1beta/models/" + MODEL + ":generateContent";
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .header("x-goog-api-key", API_KEY)
                .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .build();
        HttpResponse<String> resp = HTTP.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (resp.statusCode() != 200) {
            throw new Exception("Gemini returned HTTP " + resp.statusCode() + ": " + resp.body());
        }
        StringBuilder sb = new StringBuilder();
        Object root = Json.parse(resp.body());
        if (root instanceof Map) {
            Object cands = ((Map<?, ?>) root).get("candidates");
            if (cands instanceof List && !((List<?>) cands).isEmpty()) {
                Object cand = ((List<?>) cands).get(0);
                if (cand instanceof Map) {
                    Object content = ((Map<?, ?>) cand).get("content");
                    if (content instanceof Map) {
                        Object parts = ((Map<?, ?>) content).get("parts");
                        if (parts instanceof List) {
                            for (Object part : (List<?>) parts) {
                                if (part instanceof Map) {
                                    Object t = ((Map<?, ?>) part).get("text");
                                    if (t instanceof String) sb.append((String) t);
                                }
                            }
                        }
                    }
                }
            }
        }
        return sb.toString().trim();
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

    static String sourcesJson(List<Scored> hits) {
        StringBuilder src = new StringBuilder("[");
        for (int i = 0; i < hits.size(); i++) {
            Scored s = hits.get(i);
            if (i > 0) src.append(',');
            src.append("{\"docName\":").append(q(s.chunk.doc))
                    .append(",\"clause\":").append(q(s.chunk.clause))
                    .append(",\"page\":").append(s.chunk.page)
                    .append(",\"score\":").append(Math.round(s.score * 1000.0) / 1000.0).append('}');
        }
        return src.append(']').toString();
    }

    static void health(HttpExchange ex) throws Exception {
        send(ex, 200, "{\"status\":\"ok\",\"chunksIndexed\":" + index.chunks.size()
                + ",\"mode\":" + q(API_KEY != null ? "AI (Gemini)" : "FREE (no AI)")
                + ",\"model\":" + q(MODEL) + ",\"geminiConfigured\":" + (API_KEY != null) + "}");
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
            String g = general(question);
            send(ex, 200, "{\"answer\":" + q(g != null ? g : NOT_FOUND) + ",\"sources\":[]}");
            return;
        }

        // ---------- FREE MODE: no API key, return the best matching excerpt ----------
        if (API_KEY == null) {
            Chunk c = hits.get(0).chunk;
            String free = "Answer:\n" + c.text
                    + "\n\nCitation:\n\uD83D\uDCC4 Source: " + c.doc
                    + ", Clause " + c.clause + ", Page " + c.page + "\n";
            send(ex, 200, "{\"answer\":" + q(free) + ",\"sources\":" + sourcesJson(hits) + "}");
            return;
        }

        // ---------- AI MODE: Gemini writes a clean answer ----------
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
            answer = callGemini(SYSTEM_PROMPT, prompt);
        } catch (Exception e) {
            send(ex, 502, "{\"error\":" + q("Gemini API error: " + e.getMessage()) + "}");
            return;
        }
        if (answer.isEmpty() || answer.toLowerCase().contains(NOT_FOUND.toLowerCase())) {
            String g = general(question);
            send(ex, 200, "{\"answer\":" + q(g != null ? g : NOT_FOUND) + ",\"sources\":[]}");
            return;
        }
        send(ex, 200, "{\"answer\":" + q(answer) + ",\"sources\":" + sourcesJson(hits) + "}");
    }

    static final String PAGE =
            "<!DOCTYPE html>\n" +
            "<html lang=\"en\"><head><meta charset=\"utf-8\">\n" +
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" +
            "<title>SRM Rules Assistant</title>\n" +
            "<style>\n" +
            "*{box-sizing:border-box}\n" +
            "body{margin:0;background:#0d0b07;color:#f3e6c4;font-family:Segoe UI,Arial,sans-serif;min-height:100vh;display:flex;justify-content:center}\n" +
            ".wrap{width:100%;max-width:760px;padding:20px 16px 40px}\n" +
            "h1{color:#f5b301;font-size:1.5rem;margin:8px 0 2px}\n" +
            ".sub{color:#a8905a;font-size:.9rem;margin-bottom:18px}\n" +
            ".box{display:flex;gap:8px}\n" +
            "input{flex:1;padding:14px;border-radius:12px;border:1px solid #5a4510;background:#17130a;color:#f3e6c4;font-size:1rem;outline:none}\n" +
            "input:focus{border-color:#f5b301}\n" +
            "button{padding:14px 20px;border:0;border-radius:12px;background:linear-gradient(135deg,#f5b301,#c98a00);color:#1a1200;font-weight:700;font-size:1rem;cursor:pointer}\n" +
            "button:disabled{opacity:.5;cursor:wait}\n" +
            ".chips{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 18px}\n" +
            ".chip{background:#17130a;border:1px solid #5a4510;color:#e0c273;padding:7px 12px;border-radius:20px;font-size:.82rem;cursor:pointer}\n" +
            ".chip:hover{border-color:#f5b301}\n" +
            ".card{background:#17130a;border:1px solid #3d2f0a;border-radius:14px;padding:16px;margin-top:14px}\n" +
            ".q{color:#f5b301;font-weight:600;margin-bottom:8px}\n" +
            ".a{white-space:pre-wrap;line-height:1.55}\n" +
            ".cite{margin-top:12px;padding-top:10px;border-top:1px dashed #5a4510;color:#e0c273;font-size:.9rem;white-space:pre-wrap}\n" +
            ".err{color:#ff8a70}\n" +
            "</style></head><body><div class=\"wrap\">\n" +
            "<h1>SRM Rules Assistant</h1>\n" +
            "<div class=\"sub\">Ask about attendance, exams and college rules</div>\n" +
            "<div class=\"box\"><input id=\"q\" placeholder=\"e.g. What is the minimum attendance required?\" autofocus><button id=\"go\">Ask</button></div>\n" +
            "<div class=\"chips\">\n" +
            "<span class=\"chip\">What is the minimum attendance required?</span>\n" +
            "<span class=\"chip\">What if my attendance is below 75%?</span>\n" +
            "</div>\n" +
            "<div id=\"out\"></div>\n" +
            "</div>\n" +
            "<script>\n" +
            "var q=document.getElementById('q'),go=document.getElementById('go'),out=document.getElementById('out');\n" +
            "function ask(){\n" +
            " var text=q.value.trim(); if(!text) return;\n" +
            " go.disabled=true; go.textContent='...';\n" +
            " var card=document.createElement('div'); card.className='card';\n" +
            " card.innerHTML='<div class=\"q\"></div><div class=\"a\">Thinking...</div>';\n" +
            " card.firstChild.textContent=text; out.insertBefore(card,out.firstChild);\n" +
            " var a=card.querySelector('.a');\n" +
            " fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text})})\n" +
            " .then(function(r){return r.json()})\n" +
            " .then(function(d){\n" +
            "   if(d.error){a.className='a err';a.textContent=d.error;return;}\n" +
            "   var t=d.answer||'';\n" +
            "   var i=t.indexOf('Citation:');\n" +
            "   var main=(i>=0?t.substring(0,i):t).replace(/^Answer:\\s*/,'').trim();\n" +
            "   a.textContent=main;\n" +
            "   if(i>=0){var c=document.createElement('div');c.className='cite';c.textContent=t.substring(i+9).trim();card.appendChild(c);}\n" +
            " })\n" +
            " .catch(function(e){a.className='a err';a.textContent='Server not reachable: '+e;})\n" +
            " .then(function(){go.disabled=false;go.textContent='Ask';q.value='';q.focus();});\n" +
            "}\n" +
            "go.onclick=ask;\n" +
            "q.onkeydown=function(e){if(e.key==='Enter')ask();};\n" +
            "Array.prototype.forEach.call(document.querySelectorAll('.chip'),function(c){c.onclick=function(){q.value=c.textContent;ask();};});\n" +
            "</script></body></html>\n";

    static final String GENERAL_PROMPT =
            "You are a friendly assistant for SRM college students. The student's question is NOT covered by the "
            + "official college documents. Reply in 2-4 short, helpful, conversational sentences using general knowledge. "
            + "Start with exactly: \"Not from official SRM documents: \". "
            + "NEVER invent SRM-specific rules, numbers, dates, fees, rankings or policies. "
            + "If the question needs SRM-specific facts, say so and suggest checking with the relevant department.";

    static String general(String question) {
        if (API_KEY == null) return null;
        try {
            String g = callGemini(GENERAL_PROMPT, question);
            return g.isEmpty() ? null : g;
        } catch (Exception e) {
            return null;
        }
    }

    static void page(HttpExchange ex) throws Exception {
        String path = ex.getRequestURI().getPath();
        if (!path.equals("/") && !path.equals("/index.html")) {
            send(ex, 404, "{\"error\":\"Not found. Open / for the chat page.\"}");
            return;
        }
        byte[] b = PAGE.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
        ex.sendResponseHeaders(200, b.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(b); }
    }

    // ------------------------------ Main ------------------------------
    public static void main(String[] args) throws Exception {
        index = buildIndex();
        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/health", safe(GeminiQA::health));
        server.createContext("/reload", safe(GeminiQA::reload));
        server.createContext("/ask", safe(GeminiQA::ask));
        server.createContext("/", safe(GeminiQA::page));
        server.setExecutor(Executors.newFixedThreadPool(8));
        server.start();
        System.out.println("[INFO] Server running on http://localhost:" + PORT);
        if (API_KEY == null) {
            System.out.println("[INFO] FREE MODE active (no GEMINI_API_KEY). Answers = best matching rule text.");
        } else {
            System.out.println("[INFO] AI MODE active (Gemini model: " + MODEL + ")");
        }
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
