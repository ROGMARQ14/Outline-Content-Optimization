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

### Option 3: Using Both (Recommended)
```toml
# Add this to your Streamlit Secrets
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ANTHROPIC_API_KEY = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## ✅ Verification Steps

After adding secrets:

1. **Save** the secrets configuration
2. **Restart** your app (click "Reboot app" button)
3. **Check sidebar** in the app - should show:
   - ✅ OpenAI API Connected
   - ✅ Anthropic API Connected
4. If you see ⚠️ warnings, double-check:
   - Key names match exactly (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
   - Keys are valid and active
   - No extra spaces or quotes around values

## 💰 Cost Considerations

### OpenAI Pricing (as of 2024)
- **GPT-4o**: ~$5 per 1M input tokens, ~$15 per 1M output tokens
- Typical optimization request: $0.02 - $0.10
- **Recommended**: Start with $10 credit for testing

### Anthropic Pricing
- **Claude 3.5 Sonnet**: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- Typical optimization request: $0.01 - $0.08
- **Recommended**: Start with $10 credit for testing

### Budget Tips
1. Start with one provider to test
2. Monitor usage in API dashboards
3. Set spending limits in provider settings
4. Use Anthropic for cost-effectiveness (recommended)

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
- Verify both API keys are set (if using both)
- Test with single provider first
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