# Deploy Now — Quick Checklist

Run through this before pushing to Streamlit Cloud.  
Full details: [PUBLIC_DEPLOYMENT_PLAN.md](PUBLIC_DEPLOYMENT_PLAN.md)

---

## ✅ Done (in codebase)

- [x] Stricter rate limits (5/session, 60s between runs)
- [x] Privacy & terms in About tab
- [x] Streamlit version pinned (`<2.0`)
- [x] Python 3.10/3.11 note in requirements.txt

---

## ☐ Before You Deploy

### 1. OpenAI

- [ ] Set **budget limit** at [platform.openai.com/settings/organization/billing](https://platform.openai.com/settings/organization/billing)
- [ ] Enable **usage alerts** (e.g. 50%, 80%, 100%)

### 2. GitHub

- [ ] Code pushed to GitHub
- [ ] `secrets.toml` **not** in repo (it’s in `.gitignore`)

### 3. Streamlit Cloud

- [ ] Go to [share.streamlit.io](https://share.streamlit.io/)
- [ ] New app → select repo, branch, `app.py`
- [ ] **Advanced settings:** Python 3.10 or 3.11
- [ ] **Secrets:** add:
  ```toml
  OPENAI_API_KEY = "sk-your-key-here"
  ```
- [ ] Deploy

---

## ☐ After Deploy

- [ ] App loads
- [ ] AI Integration tab: upload image → analyze → see result
- [ ] Rate limit: run 5 analyses → see "Session limit reached"

---

## Optional

- [ ] Add `deploy_data/` if you want sample data for QC / Hi-C / Nuclei tabs without running pipelines
