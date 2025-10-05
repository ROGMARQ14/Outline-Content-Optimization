# Blog Post Optimizer - AI-Powered Content Enhancement

A comprehensive Streamlit application that leverages cutting-edge AI models (GPT-4o, Claude 3.5 Sonnet) and ML/NLP packages to optimize blog content for SEO and AI search tools.

## 🚀 Features

### Tab 1: Outline Optimizer
- **Deep Audience Research**: ICP analysis, pain points, psychological triggers
- **Search Intent Analysis**: Multi-platform search behavior insights
- **AI-Powered Optimization**: Cross-reference with Query Fan-Out analysis
- **Structured Output**: Optimized outlines with 7-12 talking points per section
- **Side-by-side Comparison**: Visual diff of original vs optimized
- **Export Options**: Download as Markdown

### Tab 2: Draft Optimizer

#### Subprocess 1: Keyword Optimization
- **Semantic Analysis**: NLP-powered relevance scoring
- **Smart Placement**: Context-aware keyword integration
- **Natural Integration**: AI-generated paragraphs maintaining tone
- **Variation Detection**: Identifies semantic similarities (>0.8 threshold)

#### Subprocess 2: AI Tool Optimization (10-Item Checklist)
1. Answer-First Introduction
2. Question-Based H2 Headings
3. Semantic Chunks (75-300 words)
4. Answer-Evidence-Context Formula
5. Direct Sentence Structures (Active Voice)
6. Informational Density (+20%)
7. Attributed Claims
8. First-Hand Experience Signals
9. Dedicated FAQ Section
10. Optimized Title & Meta

## 📋 Requirements

- Python 3.9+
- At least one AI provider API key:
  - **OpenAI API Key** (for GPT-4o models)
  - **Anthropic API Key** (for Claude models)
  - **Google API Key** (for Gemini models)
- Streamlit Cloud account (for deployment)

**Note:** You don't need all three API keys! The app intelligently detects which keys you've configured and only shows models from those providers.

## 🛠️ Local Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd blog-optimizer

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run locally
streamlit run app.py
```

## ☁️ Streamlit Cloud Deployment

### Step 1: Prepare Your Repository

1. Create a new GitHub repository
2. Add these files:
   - `app.py` (main application)
   - `requirements.txt`
   - `README.md`

### Step 2: Configure Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Select the branch and `app.py` as the main file

### Step 3: Add API Keys to Secrets

In Streamlit Cloud:
1. Go to your app settings
2. Navigate to "Secrets" section
3. Add your API keys (one or more):

**Option 1: Use all three (recommended for flexibility)**
```toml
OPENAI_API_KEY = "sk-your-openai-api-key-here"
ANTHROPIC_API_KEY = "sk-ant-your-anthropic-key-here"
GOOGLE_API_KEY = "AIzaSy-your-google-key-here"
```

**Option 2: Use only one provider**
```toml
# Just add the one(s) you have:
GOOGLE_API_KEY = "AIzaSy-your-google-key-here"
```

4. Save and deploy

**The app will automatically detect which APIs you've configured and only show those models!**

### Step 4: Deploy

Click "Deploy!" and wait for the app to build. First deployment may take 3-5 minutes.

## 🔐 Security Best Practices

- **Never commit API keys** to your repository
- **Always use Streamlit Secrets** for sensitive data
- **Rotate keys regularly** for enhanced security
- **Monitor API usage** to prevent unexpected charges

## 💡 Usage Guide

### Outline Optimizer Workflow

1. **Enter Primary Keyword**: Your target topic or keyword
2. **Paste Draft Outline**: Use Markdown format with H2/H3 structure
   ```markdown
   ## Introduction
   ## Main Topic 1
   ### Subtopic 1.1
   ### Subtopic 1.2
   ## Conclusion
   ```
3. **Add Query Fan-Out Report** (optional): Upload or paste analysis
4. **Click "Optimize Outline"**: AI performs:
   - Audience research (demographics, pain points, intent)
   - Cross-reference analysis
   - Outline enhancement with talking points
5. **Review Results**: Side-by-side comparison
6. **Download**: Export optimized outline as Markdown

### Draft Optimizer Workflow

1. **Enter Primary Keyword**: Your target SEO keyword
2. **Paste Content Draft**: Full blog post (Markdown/HTML)
3. **Add Keyword List**: Line-separated or comma-separated keywords
   ```
   dog training methods
   puppy obedience
   canine behavior
   ```
4. **Click "Optimize Draft"**: Two-stage process:
   
   **Stage 1 - Keyword Optimization:**
   - Relevance scoring for each keyword
   - Placement suggestions with context
   - AI-generated integration snippets
   
   **Stage 2 - AI Tool Optimization:**
   - 10-item checklist application
   - Full draft rewrite for AI search visibility
   - Compliance report

5. **Review Optimizations**: 
   - Keyword integration table
   - Checklist metrics
   - Final optimized draft
6. **Download**: Export as Markdown

## 🧠 Technical Architecture

### NLP Stack
- **Sentence Transformers**: Semantic similarity (all-MiniLM-L6-v2)
- **spaCy**: NLP parsing, entity extraction, syntax analysis
- **scikit-learn**: TF-IDF vectorization, cosine similarity

### AI Integration
- **OpenAI GPT-4o**: Advanced reasoning and content generation
- **Anthropic Claude 3.5 Sonnet**: High-quality optimization and analysis
- **Google Gemini**: Cost-effective processing with free tier
- **Smart Model Detection**: Automatically shows only models from configured API providers
- **Model Selection**: Choose specific models (e.g., GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash)
- **Universal Router**: Seamlessly routes requests to the correct AI provider

### Performance Optimizations
- **Model Caching**: `@st.cache_resource` for NLP models
- **Session State**: Prevents resets on file downloads
- **Progress Indicators**: Real-time feedback for long operations
- **Error Handling**: Graceful degradation with informative messages

## 🐛 Troubleshooting

### Common Issues

**"API key not found in secrets"**
- Ensure keys are added to Streamlit Cloud Secrets
- Check spelling: `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`

**"spaCy model not found"**
- First run downloads model automatically
- If persistent, redeploy app

**"Download button resets app"**
- Fixed with `create_download_link()` using base64 encoding
- Uses HTML links instead of Streamlit download buttons

**Slow performance**
- First run loads models (1-2 min one-time setup)
- Subsequent runs use cached models
- API calls vary by provider and load

### API Rate Limits

- **OpenAI**: Monitor usage at platform.openai.com
- **Anthropic**: Check console.anthropic.com
- **Google Gemini**: Free tier: 60 requests/min, view at aistudio.google.com
- Consider implementing caching for repeated requests
- **Pro Tip**: Start with Gemini's free tier for testing!

## 📊 Example Use Cases

### Use Case 1: Blog Outline Creation
**Input**: "How to train a puppy"
**Process**: 
- Audience research reveals first-time dog owners' anxiety
- Query Fan-Out includes "potty training," "crate training," "socialization"
- Optimized outline addresses pain points with actionable steps

### Use Case 2: Existing Draft Enhancement
**Input**: 2000-word draft on "Email Marketing Best Practices"
**Process**:
- Keyword analysis identifies missing terms: "email segmentation," "A/B testing"
- AI optimization converts to question-based headers
- FAQ section added for long-tail queries
- Result: 2500 words, 30% more informational density

## 🔄 Updates and Maintenance

### Updating Dependencies
```bash
pip install --upgrade streamlit anthropic openai sentence-transformers
```

### Model Updates
- Monitor Anthropic/OpenAI/Google for new model releases
- Update model strings in code (`get_available_models()` function):
  - OpenAI: `"gpt-4o"` (or latest)
  - Anthropic: `"claude-3-5-sonnet-20241022"` (or latest)
  - Google: `"gemini-2.0-flash-exp"` (or latest)
- Add new models to the models dictionary for each provider

## 📝 License

This project is provided as-is for educational and commercial use.

## 🤝 Support

For issues, feature requests, or questions:
1. Check this README first
2. Review Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
3. Check API provider docs:
   - [OpenAI API Docs](https://platform.openai.com/docs)
   - [Anthropic API Docs](https://docs.anthropic.com)
   - [Google Gemini API Docs](https://ai.google.dev/docs)

## 🎯 Roadmap

Future enhancements:
- [ ] PDF export functionality
- [ ] Real-time SerpAPI integration for live search data
- [ ] Multi-language support
- [ ] Content scoring dashboard
- [ ] Historical version tracking
- [ ] Bulk processing mode

---

**Built with ❤️ using Streamlit, OpenAI, Anthropic, and cutting-edge NLP**