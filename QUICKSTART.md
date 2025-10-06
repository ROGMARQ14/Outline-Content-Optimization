# Quick Start Guide - Deploy in 10 Minutes

Follow this checklist to deploy your Blog Post Optimizer successfully.

## ✅ Pre-Deployment Checklist

### Step 0: Python Version (IMPORTANT!) ⚠️

- [ ] Verify you're using **Python 3.11** (not 3.12)
- [ ] If using Python 3.12, see [PYTHON_VERSION_GUIDE.md](PYTHON_VERSION_GUIDE.md) first
- [ ] Your repository should include `.python-version` file with "3.11"

**Why?** spaCy (required for NLP) has compatibility issues with Python 3.12. Python 3.11 works perfectly.

The `.python-version` file is included in the repository - just make sure it's uploaded to GitHub!

### Step 1: Get Your Files Ready (2 minutes)

- [ ] Create a new folder for your project
- [ ] Add `app.py` (main application file)
- [ ] Add `requirements.txt` (dependencies)
- [ ] Add `.python-version` file with content: `3.11`
- [ ] **CRITICAL:** Verify `requirements.txt` has spaCy model URL on last line:
  ```txt
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
  ```
- [ ] Create `.streamlit/config.toml` (optional, for better UI)

### Step 2: Get API Key(s) (3 minutes)

You need **at least one** API key. Choose based on your needs:

#### Option A: Free Testing (Recommended for Beginners)
- [ ] Get Google Gemini API key (FREE):
  1. Go to [aistudio.google.com](https://aistudio.google.com)
  2. Sign in with Google account
  3. Click "Get API Key" 
  4. Copy key (starts with `AIzaSy...`)

#### Option B: Best Quality/Cost Balance
- [ ] Get Anthropic API key:
  1. Go to [console.anthropic.com](https://console.anthropic.com)
  2. Sign up for account
  3. Add billing info ($5 minimum)
  4. Create API key
  5. Copy key (starts with `sk-ant-api03...`)

#### Option C: Maximum Capability
- [ ] Get OpenAI API key:
  1. Go to [platform.openai.com](https://platform.openai.com)
  2. Sign up for account
  3. Add billing info ($10 minimum recommended)
  4. Create API key
  5. Copy key (starts with `sk-proj...`)

#### Option D: All Three (Maximum Flexibility)
- [ ] Get all three keys from above options

**Important:** Save your key(s) in a safe place immediately! Some are only shown once.

### Step 3: Create GitHub Repository (2 minutes)

- [ ] Create new repository on GitHub
- [ ] Name it (e.g., "blog-optimizer")
- [ ] Make it Public or Private (your choice)
- [ ] Don't initialize with README (you have files already)
- [ ] Copy repository URL

### Step 4: Push Code to GitHub (1 minute)

```bash
cd your-project-folder
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

- [ ] Verify files are on GitHub
- [ ] Check that `requirements.txt` uploaded correctly

### Step 5: Deploy to Streamlit Cloud (2 minutes)

- [ ] Go to [share.streamlit.io](https://share.streamlit.io)
- [ ] Click "New app"
- [ ] Sign in with GitHub
- [ ] Grant Streamlit access to your repositories
- [ ] Select your repository
- [ ] Select branch: `main`
- [ ] Select main file: `app.py`
- [ ] Click "Deploy!" (but DON'T close the window yet!)

### Step 6: Configure Secrets (1 minute)

**Before the app fully deploys:**

- [ ] Click "Advanced settings"
- [ ] Click "Secrets" tab
- [ ] Add your API key(s) in TOML format:

**If you have Google Gemini only:**
```toml
GOOGLE_API_KEY = "AIzaSy-your-actual-key-here"
```

**If you have Anthropic only:**
```toml
ANTHROPIC_API_KEY = "sk-ant-api03-your-actual-key-here"
```

**If you have OpenAI only:**
```toml
OPENAI_API_KEY = "sk-proj-your-actual-key-here"
```

**If you have multiple keys:**
```toml
GOOGLE_API_KEY = "AIzaSy-your-key-here"
ANTHROPIC_API_KEY = "sk-ant-api03-your-key-here"
OPENAI_API_KEY = "sk-proj-your-key-here"
```

- [ ] Click "Save"
- [ ] Close settings window

### Step 7: Wait for Deployment (3-5 minutes)

- [ ] Watch the deployment logs
- [ ] Look for "Your app is live!"
- [ ] **First deployment takes 3-5 minutes** (downloading models)
- [ ] Subsequent loads will be fast

### Step 8: Verify It's Working (1 minute)

Once deployed, check:

- [ ] App loads without errors
- [ ] Sidebar shows "✅ API Connected" for your configured key(s)
- [ ] Model selector dropdown has options
- [ ] Two tabs visible: "Outline Optimizer" and "Draft Optimizer"

### Step 9: Test Basic Functionality (2 minutes)

**Quick Test:**
- [ ] Go to "Outline Optimizer" tab
- [ ] Enter keyword: "test topic"
- [ ] Enter simple outline:
  ```markdown
  ## Introduction
  ## Main Point
  ## Conclusion
  ```
- [ ] Click "Optimize Outline"
- [ ] Wait for results (should take 10-30 seconds)
- [ ] Verify you see optimized outline
- [ ] Check download link works

---

## 🎉 Success!

If all above steps complete successfully, you're ready to use the app!

## 📋 What to Do Next

### For Testing (Free Tier)
1. Use Gemini models for unlimited testing
2. Experiment with different prompts
3. Test both tabs (Outline & Draft)
4. Export results as Markdown

### For Production Use
1. Add billing to your chosen provider(s)
2. Monitor API usage in provider dashboard
3. Set spending limits
4. Create workflow for your content team

### Customize the App
1. Update model selection in code
2. Adjust prompt templates
3. Modify UI colors in config.toml
4. Add more features as needed

---

## ⚠️ Common First-Time Issues

### Issue: "spaCy model not found"
**Fix:** Make sure last line of `requirements.txt` is the spaCy model URL. Redeploy.

### Issue: "No API keys configured"
**Fix:** Add at least one API key to Streamlit Secrets. Reboot app.

### Issue: Shows "API Connected" but doesn't work
**Fix:** Verify your API key is actually valid on the provider's website. Check for typos.

### Issue: App takes forever to load
**Fix:** This is normal for first deployment (3-5 min). Be patient. Check logs.

---

## 🆘 Need Help?

- **Read:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions
- **Read:** [README.md](README.md) for full documentation
- **Check:** Streamlit Cloud logs (click "Manage app" → "Logs")
- **Ask:** Streamlit Community Forum at [discuss.streamlit.io](https://discuss.streamlit.io)

---

## 💰 Cost Estimates

### Free Tier (Google Gemini)
- **Cost:** $0
- **Limits:** 60 requests/minute
- **Good for:** Testing, learning, small-scale use

### Budget Setup (Gemini + Anthropic)
- **Cost:** ~$5-10/month
- **Volume:** ~200-500 optimizations
- **Good for:** Freelancers, small teams

### Professional Setup (All Three)
- **Cost:** ~$20-50/month
- **Volume:** ~1000-2000 optimizations
- **Good for:** Agencies, content teams

---

## ✅ Deployment Complete!

Time to optimize some blog posts! 🚀

**Bookmark your app URL:** `https://your-app-name.streamlit.app`

Share it with your team and start creating better content!