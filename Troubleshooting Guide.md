# Troubleshooting Guide

## 🔥 Common Issues & Solutions

### Issue 1: "spaCy model not found" Error

**Error Message:**
```
OSError: [E050] Can't find model 'en_core_web_sm'
```

**Root Cause:**
Streamlit Cloud can't download spaCy models using subprocess commands.

**✅ Solution:**

Make sure your `requirements.txt` includes the direct spaCy model download URL:

```txt
streamlit>=1.28.0
anthropic>=0.18.0
openai>=1.12.0
google-generativeai>=0.3.0
sentence-transformers>=2.2.2
spacy>=3.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
torch>=2.0.0
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

**The last line is critical!** It downloads the spaCy model directly during deployment.

After updating `requirements.txt`:
1. Push changes to GitHub
2. Streamlit Cloud will auto-redeploy
3. Wait 3-5 minutes for deployment

---

### Issue 2: False API Connection Status

**Problem:**
Sidebar shows "✅ API Connected" even though you didn't provide that API key.

**Root Cause:**
Empty strings or whitespace in Streamlit Secrets being treated as valid keys.

**✅ Solution:**

The app now validates API keys with these checks:
- Key must exist in secrets
- Key must not be empty
- Key must be at least 10 characters long

**How to verify your secrets are correct:**

1. Go to Streamlit Cloud app settings
2. Click "Secrets" tab
3. Ensure format is:
   ```toml
   OPENAI_API_KEY = "sk-proj-actual-key-here"
   GOOGLE_API_KEY = "AIzaSy-actual-key-here"
   ```
4. **No empty lines** or keys without values
5. **No quotes inside quotes**: `"sk-..."` not `""sk-...""` 
6. Save and reboot app

**Common mistakes:**
```toml
# ❌ WRONG - Empty value
ANTHROPIC_API_KEY = ""

# ❌ WRONG - Just whitespace
ANTHROPIC_API_KEY = "   "

# ❌ WRONG - Placeholder text
ANTHROPIC_API_KEY = "your-key-here"

# ✅ CORRECT - Actual key or omit entirely
ANTHROPIC_API_KEY = "sk-ant-api03-actual-key-Avf3jP9..."

# ✅ CORRECT - Don't include keys you don't have
# (Just omit the line entirely)
```

---

### Issue 3: App Won't Start - "No API keys configured"

**Error Message:**
```
❌ No API keys configured! Please add at least one API key to Streamlit Secrets.
```

**Root Cause:**
No valid API keys found in secrets.

**✅ Solution:**

1. Add **at least one** valid API key to Streamlit Secrets
2. You don't need all three - just one is enough
3. Minimum viable setup:

```toml
GOOGLE_API_KEY = "AIzaSyDxxxxxxxxxxxxxxxxxxxxx"
```

This is enough to start using the app with Gemini models!

---

### Issue 4: Model Selection Dropdown is Empty

**Problem:**
The "Select AI Model" dropdown has no options.

**Root Cause:**
No valid API keys detected, or all keys are invalid.

**✅ Solution:**

Check the "API Connection Status" section in the sidebar:
- If all show ⚠️ warnings → No valid keys configured
- Fix by adding at least one valid key
- After adding key, reboot app
- Model dropdown will populate automatically

**Debugging steps:**
1. Check secrets format (no typos in key names)
2. Verify API keys are valid on provider websites
3. Check for extra spaces or quotes
4. Ensure keys are at least 10 characters

---

### Issue 5: "API Error" When Running Optimization

**Error Message:**
```
❌ OpenAI API Error: Incorrect API key provided
```

**Root Cause:**
API key is invalid, expired, or formatted incorrectly.

**✅ Solution:**

1. **Verify key on provider platform:**
   - OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Anthropic: [console.anthropic.com](https://console.anthropic.com)
   - Google: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

2. **Regenerate key** if needed

3. **Update Streamlit Secrets** with new key

4. **Reboot app**

**Common API key format issues:**
- Extra spaces before/after key
- Missing prefix (sk-proj for OpenAI, sk-ant for Anthropic, AIzaSy for Google)
- Truncated key (copied only part of it)

---

### Issue 6: Slow First Load

**Observation:**
App takes 2-3 minutes to load on first run.

**Root Cause:**
This is **normal behavior**! The app needs to:
1. Download spaCy model (if not cached)
2. Download sentence-transformers model (~90MB)
3. Load NLP models into memory

**✅ Not an issue - expected behavior**

**Subsequent loads are fast** because models are cached.

To verify it's working:
- Check build logs in Streamlit Cloud
- Look for "Loading NLP models..." message
- Wait patiently - this is one-time setup

---

### Issue 7: Torch/PyTorch Installation Issues

**Error Message:**
```
ERROR: Could not find a version that satisfies the requirement torch
```

**Root Cause:**
Platform compatibility or dependency conflicts.

**✅ Solution:**

Update `requirements.txt` to specify CPU-only PyTorch:

```txt
streamlit>=1.28.0
anthropic>=0.18.0
openai>=1.12.0
google-generativeai>=0.3.0
sentence-transformers>=2.2.2
spacy>=3.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.0.0
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

The `--extra-index-url` line helps Streamlit Cloud find CPU-optimized PyTorch.

---

### Issue 8: App Works Locally But Fails on Streamlit Cloud

**Common causes:**

1. **Secrets not configured** on Streamlit Cloud
   - Local: Uses environment variables or .streamlit/secrets.toml
   - Cloud: Must configure in Streamlit Cloud UI

2. **Subprocess commands** don't work in cloud
   - Fixed in latest version (uses direct download URL)

3. **File paths** might be different
   - Use relative paths, not absolute

4. **Write permissions** - cloud is read-only
   - Don't try to write files to disk
   - Use session state or memory instead

**✅ Solution:**
- Always test deployment on Streamlit Cloud
- Check logs in "Manage app" section
- Use st.error() to debug issues

---

## 🔍 How to Debug

### Step 1: Check Streamlit Cloud Logs

1. Go to your app on Streamlit Cloud
2. Click "Manage app" (bottom right)
3. Click "Logs" tab
4. Look for error messages (usually in red)

### Step 2: Verify Secrets Configuration

```python
# Add this temporarily to your app.py to debug:
st.write("Debugging secrets:")
openai_key, anthropic_key, google_key = get_api_keys()
st.write(f"OpenAI key length: {len(openai_key) if openai_key else 0}")
st.write(f"Anthropic key length: {len(anthropic_key) if anthropic_key else 0}")
st.write(f"Google key length: {len(google_key) if google_key else 0}")
```

**Don't display actual keys!** Just show lengths to verify they exist.

### Step 3: Test API Calls Manually

Add a test button to verify API connectivity:

```python
if st.button("Test OpenAI API"):
    result = call_openai("Say 'API working!'", max_tokens=10)
    st.write(result)
```

---

## 📝 Pre-Deployment Checklist

Before deploying to Streamlit Cloud:

- [ ] `requirements.txt` includes spaCy model download URL
- [ ] At least one API key added to Streamlit Secrets
- [ ] API key format verified (no extra spaces/quotes)
- [ ] No hardcoded API keys in code
- [ ] Tested locally first
- [ ] Removed any debug code
- [ ] Committed all changes to GitHub

---

## 🆘 Still Having Issues?

If none of the above solutions work:

1. **Check Streamlit Community Forum:**
   - [discuss.streamlit.io](https://discuss.streamlit.io)
   - Search for similar issues

2. **Review provider status pages:**
   - OpenAI: [status.openai.com](https://status.openai.com)
   - Anthropic: Check Twitter @AnthropicAI
   - Google: [status.cloud.google.com](https://status.cloud.google.com)

3. **Verify account status:**
   - Check if API quota exceeded
   - Verify billing is active
   - Ensure account is in good standing

4. **Create minimal reproduction:**
   - Strip down to simplest possible code
   - Test with just one API call
   - Isolate the problematic component

---

## ✅ Success Indicators

Your app is working correctly when you see:

1. ✅ At least one "API Connected" in sidebar
2. ✅ Model selector dropdown populated
3. ✅ "Loading NLP models..." completes without error
4. ✅ Test optimization completes successfully
5. ✅ Download links work without page refresh

If all above are true → **You're all set!** 🎉