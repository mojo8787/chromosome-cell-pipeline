# Security and Stability

## Security Measures

### API Keys and Secrets
- **Never stored in code or config** — API keys are read from environment variables or Streamlit secrets only
- **`secrets.toml` is gitignored** — Local secrets file is never committed
- **Password input** — User-entered keys use `type="password"` (masked in UI)
- **Temporary env usage** — When running VLM from the app, the key is set in `os.environ` only during the API call and restored in a `finally` block

### File Upload
- **Path traversal prevention** — Uploaded filenames are sanitized; only the basename is used and paths are validated to stay inside the temp directory
- **File type restriction** — Streamlit `file_uploader` restricts to PNG, JPG, TIFF
- **File size limit** — 50 MB per file to prevent DoS via large uploads
- **Upload limit** — Maximum 10 images per analysis run
- **Temporary storage** — Files are written to `tempfile.TemporaryDirectory()` and automatically deleted after use

### Subprocess
- **No shell execution** — `subprocess.run()` uses a list of arguments (no `shell=True`)
- **Fixed script paths** — Only predefined pipeline scripts are executed

### Configuration
- **`yaml.safe_load`** — Config is loaded with `safe_load` to prevent arbitrary object deserialization
- **Path handling** — All file paths are derived from config or `ROOT`; no user-controlled paths for file reads

### HTML and XSS
- **Static CSS/HTML** — Custom UI uses static, non-user content
- **Streamlit escaping** — `st.write()` escapes user content by default; VLM descriptions are safe

---

## Stability Measures

### Error Handling
- **VLM loop** — Each image is processed in try/except; failures are logged, processing continues
- **Empty state handling** — App checks for file existence before reading; shows informative messages when data is missing
- **Secrets fallback** — `st.secrets.get()` wrapped in try/except for environments without secrets

### Dependencies
- **Pinned versions** — `requirements.txt` specifies minimum versions for reproducibility
- **No eval/exec of user input** — No dynamic code execution from user data

### Resource Limits
- **Image count** — Configurable `max_images` (default 5) for pipeline overlays
- **Upload limit** — 10 images per upload batch

---

## Recommendations

1. **Deployment** — Use Streamlit secrets for production; avoid exposing API key input to end users if you pay for usage
2. **Updates** — Keep dependencies updated; run `pip install -r requirements.txt --upgrade` periodically
3. **Monitoring** — Monitor API usage and costs when using server-side keys
4. **Data** — Uploaded images are processed in memory and temp files; they are not persisted unless you explicitly save outputs
