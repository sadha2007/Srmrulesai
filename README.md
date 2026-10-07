# SRM Rules Assistant

A single-file Java app that answers student questions about college rules (attendance, exams, circulars) using your own documents, and always shows the clause it used.

No frameworks, no build tools, no external libraries.

## TL;DR

```bash
java hooo.java
```

Then open **http://localhost:8080**

---

## Features

- Chat web page built in (dark amber/gold, mobile friendly)
- Answers only from your documents, with a citation (document, clause, page)
- Works with or without an AI key
  - **Free mode:** no key, returns the best matching rule text
  - **AI mode:** free Gemini key, returns a clean, student-friendly answer
- Out-of-scope questions get a short general answer, clearly labeled "Not from official SRM documents". The bot never invents SRM rules
- Hot reload of documents, no restart needed

## Requirements

| Item | Needed |
|---|---|
| Java | 11 or higher (17 recommended). Check with `java -version` |
| Gemini API key | Optional. Free from https://aistudio.google.com |
| Internet | Only for AI mode |

## Folder structure

```
srm-qa/
├── hooo.java          <- the whole app
└── documents/         <- your rules, as .txt files
    ├── attendance.txt
    └── exam-rules.txt
```

The `documents` folder is created automatically on first run if it is missing. It must sit **next to the .java file** (or set `DOCS_DIR`).

## Quick start

1. Put `hooo.java` in a folder.
2. Create a `documents` folder beside it and add `.txt` files.
3. (Optional) Set your API key, see below.
4. Run:
   ```bash
   java hooo.java
   ```
5. Open http://localhost:8080 and ask a question.

Expected console output:

```
[INFO] Indexed 12 chunks from 'documents'
[INFO] Server running on http://localhost:8080
[INFO] AI MODE active (Gemini model: gemini-3.5-flash-lite)
```

Keep that terminal open. Closing it stops the server.

## Setting the API key (AI mode)

Get a free key at https://aistudio.google.com, then set it **in the same terminal** before running:

| System | Command |
|---|---|
| Windows CMD | `set GEMINI_API_KEY=your_key` |
| PowerShell | `$env:GEMINI_API_KEY="your_key"` |
| Mac / Linux | `export GEMINI_API_KEY=your_key` |

Without a key the app runs in **free mode** automatically.

> **Security:** never commit your key to GitHub or share the file with a key written inside it. Prefer the environment variable. If a key is ever exposed, delete it in AI Studio and create a new one.

## Configuration

All settings are environment variables. All are optional.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | none | Enables AI mode (`GOOGLE_API_KEY` also works) |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini model to use |
| `DOCS_DIR` | `documents` | Folder with your rules files |
| `PORT` | `8080` | Server port |
| `GEMINI_BASE_URL` | Google's default | Override the API endpoint |

Example: `set PORT=9090` then `java hooo.java`.

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Chat web page |
| `/ask` | POST | Ask a question |
| `/health` | GET | Status, chunk count, current mode and model |
| `/reload` | GET | Re-read the `documents` folder |

**Ask from the command line**

PowerShell:

```powershell
(Invoke-RestMethod -Uri http://localhost:8080/ask -Method Post -ContentType "application/json" -Body '{"question":"What is the minimum attendance required?"}').answer
```

curl:

```bash
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" -d '{"question":"What is the minimum attendance required?"}'
```

**Response**

```json
{
  "answer": "Answer: ...\n\nCitation: Source: attendance, Clause 4.1, Page 1",
  "sources": [{"docName": "attendance", "clause": "4.1", "page": 1, "score": 0.42}]
}
```

## Preparing your documents

Answer and citation quality depend mostly on how the `.txt` files are written.

1. **Number your clauses** at the start of a line, for example `4.2 Attendance shortage ...`. The app detects these and uses them as the clause in the citation.
2. **Add page breaks** with a form-feed character (Ctrl+L in many editors) between pages so page numbers are correct. Otherwise everything counts as page 1.
3. **One topic per clause.** Smaller, focused clauses retrieve better.
4. **Use .txt or .md.** PDFs need extra setup (see below).
5. After adding or editing files, open http://localhost:8080/reload

Example `documents/attendance.txt`:

```
4.1 Minimum attendance
Students must maintain at least 75 percent attendance in each course
to be eligible for the end semester examination.

4.2 Attendance shortage
Students with attendance between 65 and 74 percent may apply for
condonation with a medical certificate.
```

### PDF support (optional)

Easiest: convert the PDF to `.txt`. Or download `pdfbox-app-3.x.jar` from Apache PDFBox and run:

```bash
java -cp pdfbox-app-3.0.x.jar hooo.java
```

## How it works

1. On start, documents are split into chunks by clause and page.
2. Chunks are indexed with TF-IDF.
3. A question retrieves the top 5 matching chunks.
4. **AI mode:** the chunks go to Gemini with strict rules: answer only from the excerpts, cite the clause, never fabricate.
5. **Free mode:** the best chunk is returned with its citation.
6. If nothing matches, AI mode gives a short general answer labeled as not official. Free mode says the information is unavailable.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Indexed 0 chunks` | `documents` is empty, has the wrong file type, or is in the wrong place. Check with `dir documents`, then open `/reload` |
| Browser shows `404 Not Found` at `/` | An old version is still running. Press Ctrl+C, replace the file, run again |
| `localhost refused to connect` | The server is not running. Start it and keep the window open |
| `Address already in use` | Another instance is running. Close it, or set a different `PORT` |
| Gemini `404` model error | The model was retired. Set `GEMINI_MODEL` to a current one |
| Gemini `429` | Free-tier limit reached. Wait a minute or try another model |
| Gemini `400` / `403` | Invalid key. Create a new one |
| Still in free mode after setting key | The key was set in a different terminal. Set it again in the same one |
| `/ask` shows `Use POST` in the browser | Expected. Use the chat page or a POST request |

Tip: `/health` shows the real mode and model the server is running.

## Limitations

- Keyword-based retrieval (TF-IDF). It will not catch every paraphrase
- Free Gemini tiers are rate limited, and free-tier data may be used by Google to improve its products, so avoid sending private student data
- Documents are held in memory, which is fine for a few hundred pages
- No login. Do not expose the port to the public internet as is

## License

Use and modify freely for your own project.
