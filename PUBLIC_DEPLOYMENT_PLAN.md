# Public Deployment Plan — Chromosome + Cell Pipeline

Best-practices checklist and plan for deploying this app publicly. Based on Streamlit, OpenAI, and general security guidelines.

---

## Phase 1: Pre-Deployment Checklist

### 1.1 Secrets & Security

| Task | Status | Notes |
|------|--------|-------|
| API key in `secrets.toml` (local) or Cloud Secrets | ☐ | Never in code or git |
| `secrets.toml` in `.gitignore` | ✓ | Already excluded |
| No hardcoded credentials | ☐ | Audit `app.py` and scripts |
| HTTPS only | ✓ | Streamlit Cloud uses HTTPS by default |

### 1.2 Code & Dependencies

| Task | Status | Notes |
|------|--------|-------|
| Pin versions in `requirements.txt` | ☐ | Reduces build failures |
| Python 3.10 or 3.11 | ☐ | Set in Cloud app settings |
| No untrusted pickle in `st.cache_data` | ☐ | Session state holds user uploads — acceptable for in-memory only |

### 1.3 Cost & Rate Limiting

| Task | Status | Notes |
|------|--------|-------|
| Rate limits configured in `config.yaml` | ✓ | 10/session, 30s between runs |
| OpenAI usage limits set | ☐ | Set in OpenAI dashboard |
| OpenAI budget alerts | ☐ | Configure in billing settings |

---

## Phase 2: Recommended Config for Public Use

### 2.1 Stricter Rate Limits (`config.yaml`)

For public deployment, use tighter limits:

```yaml
rate_limit:
  max_analyses_per_session: 5
  min_seconds_between_analyses: 60
  warning_at_remaining: 1
```

### 2.2 OpenAI Account Settings

1. Go to [platform.openai.com/settings/organization/billing](https://platform.openai.com/settings/organization/billing)
2. Set **Hard limit** (e.g. $50/month)
3. Enable **Usage alerts** (e.g. at 50%, 80%, 100%)

---

## Phase 3: Legal & User Communication

### 3.1 Privacy Notice

Add a short notice about data handling. Suggested content for the About tab:

- **Data processing:** Images are sent to OpenAI for analysis. OpenAI’s privacy policy applies.
- **Retention:** No images or descriptions are stored on our servers; session data is cleared when you leave.
- **Purpose:** AI phenotype analysis of microscopy images.

### 3.2 Terms / Disclaimer

Add to the About tab:

- **Intended use:** For research and educational purposes.
- **No warranty:** Use at your own risk; results are not medical or clinical advice.
- **User responsibility:** Use only images you have the right to share and analyze.

### 3.3 Third-Party Attribution

- Open source dependencies (cooltools, StarDist, etc.)
- Data sources (BBBC039, etc.) if shown in the app

---

## Phase 4: Deployment Steps (Streamlit Community Cloud)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. New app → select repo, branch, `app.py`
4. **Advanced settings:**
   - Python: 3.10 or 3.11
   - Secrets: paste TOML with `OPENAI_API_KEY`

### Step 3: Post-Deploy Verification

- [ ] App loads
- [ ] QC, Hi-C, Nuclei tabs work (with deploy_data)
- [ ] AI Integration tab accepts uploads
- [ ] Analysis runs and shows results
- [ ] Rate limit message appears when limit reached

---

## Phase 5: Post-Deployment Monitoring

### 5.1 Regular Checks

| Check | Frequency | Action |
|-------|-----------|--------|
| OpenAI usage | Weekly | Billing dashboard |
| App uptime | Ongoing | Streamlit Cloud status |
| Error logs | After deploy | Cloud logs in app settings |

### 5.2 Abuse Mitigation

- **Session reset:** Users can refresh to get a new session limit. Acceptable for demos.
- **Upgrade:** For stricter limits, add IP-based rate limiting via a reverse proxy (e.g. Cloudflare, nginx).

---

## Phase 6: Optional Enhancements

| Enhancement | Effort | Benefit |
|------------|--------|---------|
| Add Privacy/Terms section to About tab | Low | Legal clarity |
| Tighter default rate limits | Low | Lower cost risk |
| OpenAI usage tracking in app | Medium | Visibility into costs |
| Optional auth (e.g. password gate) | Medium | More control |
| IP-based rate limiting | High | Stronger abuse protection |

---

## Summary Checklist

Before going live:

- [ ] Secrets configured in Cloud (not in code)
- [ ] Rate limits tightened for public use
- [ ] OpenAI budget + alerts set
- [ ] Privacy notice and disclaimer added
- [ ] App tested end-to-end after deploy
- [ ] Monitoring plan in place

---

## References

- [Streamlit Deployment Concepts](https://docs.streamlit.io/deploy/concepts)
- [Streamlit Secrets Management](https://docs.streamlit.io/deploy/concepts/secrets)
- [OpenAI Production Best Practices](https://platform.openai.com/docs/guides/production-best-practices)
- [Streamlit Security Reminders](https://docs.streamlit.io/develop/concepts/connections/security-reminders)
