# Streamlit Cloud Secrets Configuration Guide

## 📍 Where to Add Secrets

1. **Deploy your app** to Streamlit Cloud first
2. Go to your app dashboard at [share.streamlit.io](https://share.streamlit.io)
3. Click on your app name
4. Click the **"Settings"** button (⚙️) in the bottom right
5. Navigate to the **"Secrets"** tab

## 🔑 Required API Keys

### Option 1: Using OpenAI Only
```toml
# Add this to your Streamlit Secrets
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Get your OpenAI API Key:**
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign in or create account
3. Navigate to API Keys section
4. Click "Create new secret key"
5. Copy and save immediately (shown once only!)

**Available OpenAI Models:**
- GPT-4o - Latest, most capable model
- GPT-4o Mini - Fast & efficient for most tasks
- GPT-4 Turbo - Balanced performance

### Option 2: Using Anthropic Only
```toml
# Add this to your Streamlit Secrets
ANTHROPIC_API_KEY = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Get your Anthropic API Key:**
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign in or create account
3. Navigate to API Keys
4. Click "Create Key"
5. Copy and save the key

**Available Anthropic Models:**
- Claude 3.5 Sonnet - Recommended for quality/speed balance
- Claude 3.5 Haiku - Fastest, most cost-effective Claude
- Claude 3 Opus - Most capable (older generation)

### Option 3: Using Anthropic + OpenAI
```toml
# Add this to your Streamlit Secrets
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ANTHROPIC_API_KEY = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Option 4: Using Google Gemini Only
```toml
# Add this to your Streamlit Secrets
GOOGLE_API_KEY = "AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Get your Google API Key:**
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with Google account
3. Click "Get API Key"
4. Create API key in new or existing project
5. Copy and save the key

**Available Gemini Models:**
- Gemini 2.0 Flash (Experimental) - Fastest, most cost-effective
- Gemini 1.5 Pro - Balanced performance
- Gemini 1.5 Flash - Fast & cost-effective

### Option 5: Using All Three (Maximum Flexibility - Recommended)
```toml
# Add this to your Streamlit Secrets
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ANTHROPIC_API_KEY = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GOOGLE_API_KEY = "AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## ✅ Verification Steps

After adding secrets:

1. **Save** the secrets configuration
2. **Restart** your app (click "Reboot app" button)
3. **Check sidebar** in the app - should show:
   - ✅ OpenAI API Connected (if configured)
   - ✅ Anthropic API Connected (if configured)
   - ✅ Google Gemini API Connected (if configured)
4. The **Model Selector** will automatically show only models from providers with configured API keys
5. If you see ⚠️ warnings, double-check:
   - Key names match exactly (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`)
   - Keys are valid and active
   - No extra spaces or quotes around values

## 🎯 How Smart Model Detection Works

The app automatically:
- ✅ **Detects** which API keys are present in your secrets
- ✅ **Shows only** models from configured providers
- ✅ **Hides** unavailable providers completely
- ✅ **Prevents errors** by not offering models you can't use

**Examples:**
- Only have OpenAI key? → See only GPT models
- Have Anthropic + Google? → See Claude and Gemini models (no GPT)
- Have all three? → Full selection of all models

You can mix and match any combination of providers!

## 💰 Cost Considerations

### OpenAI Pricing (as of 2024)
- **GPT-4o**: ~$5 per 1M input tokens, ~$15 per 1M output tokens
- Typical optimization request: $0.02 - $0.10
- **Recommended**: Start with $10 credit for testing

### Anthropic Pricing
- **Claude 3.5 Sonnet**: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- Typical optimization request: $0.01 - $0.08
- **Recommended**: Start with $10 credit for testing

### Google Gemini Pricing
- **Gemini 1.5 Pro**: Free tier available (60 requests/minute)
- **Gemini 2.0 Flash**: ~$0.075 per 1M input tokens, ~$0.30 per 1M output tokens
- Typical optimization request: $0.001 - $0.03 (most cost-effective)
- **Recommended**: Start with free tier for testing

### Budget Tips
1. **Start with Google Gemini** (free tier) for initial testing
2. Monitor usage in API dashboards
3. Set spending limits in provider settings
4. Use Gemini for cost-effectiveness (cheapest)
5. Use Anthropic for quality/cost balance
6. Use OpenAI for specific GPT-4o features

## 🔒 Security Best Practices

### DO ✅
- Use Streamlit Secrets for all API keys
- Rotate keys every 90 days
- Set spending limits in API provider dashboards
- Monitor usage regularly
- Use read-only keys if available

### DON'T ❌
- Never commit API keys to GitHub
- Never share keys in public channels
- Never hardcode keys in app.py
- Never use production keys for testing

## 🧪 Testing Your Setup

### Quick Test Process

1. **Deploy app** with secrets configured
2. **Open app** in browser
3. **Navigate to "Outline Optimizer" tab**
4. **Enter test data**:
   - Keyword: "test keyword"
   - Outline: "## Test Heading\n\n## Another Heading"
5. **Click "Optimize Outline"**
6. **Check for**:
   - No API errors
   - Progress indicators working
   - Results displayed
   - Download link appears

If all above work ✅ → Setup complete!

## 🐛 Troubleshooting

### Error: "API key not found in secrets"
**Solution**: 
- Go to Settings → Secrets
- Verify key names match exactly
- Check for typos
- Reboot app after saving

### Error: "Invalid API key"
**Solution**:
- Verify key is active in provider dashboard
- Regenerate key if needed
- Update secrets with new key
- Reboot app

### Error: Rate limit exceeded
**Solution**:
- Wait a few minutes
- Check API dashboard for usage
- Upgrade API plan if needed
- Switch to alternative provider

### App loads but features don't work
**Solution**:
- Check browser console (F12) for errors
- Verify at least one API key is configured
- The model selector should show available models automatically
- If no models appear, recheck secrets configuration
- Check Streamlit logs in app dashboard

## 📱 Mobile Access

The app works on mobile browsers, but desktop recommended for:
- Better textarea editing
- Side-by-side comparisons
- Download functionality

## 🔄 Updating Secrets

To change API keys:
1. Go to app Settings → Secrets
2. Update the key values
3. Click "Save"
4. Click "Reboot app"
5. Refresh browser

No need to redeploy the entire app!

## 📞 Getting Help

**Streamlit Issues**:
- [Streamlit Community Forum](https://discuss.streamlit.io)
- [Streamlit Docs - Secrets](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)

**API Issues**:
- OpenAI: [platform.openai.com/docs](https://platform.openai.com/docs)
- Anthropic: [docs.anthropic.com](https://docs.anthropic.com)

---

**You're all set! 🎉**

Start optimizing your blog content with AI-powered insights!