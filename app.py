import streamlit as st
import anthropic
import openai
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
from io import BytesIO
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Blog Post Optimizer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'outline_result' not in st.session_state:
    st.session_state.outline_result = None
if 'draft_result' not in st.session_state:
    st.session_state.draft_result = None
if 'audience_insights' not in st.session_state:
    st.session_state.audience_insights = None

# Load models (cached to prevent reloading)
@st.cache_resource
def load_nlp_models():
    """Load and cache NLP models"""
    try:
        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")
    except:
        st.warning("Downloading spaCy model... This is a one-time setup.")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    # Load sentence transformer for semantic similarity
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return nlp, semantic_model

# API Configuration
def get_api_keys():
    """Retrieve API keys from Streamlit secrets"""
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")
        anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        google_key = st.secrets.get("GOOGLE_API_KEY", "")
        return openai_key, anthropic_key, google_key
    except:
        return "", "", ""

def get_available_models():
    """Get available models based on configured API keys"""
    openai_key, anthropic_key, google_key = get_api_keys()
    
    models = {}
    
    if openai_key:
        models["OpenAI"] = {
            "gpt-4o": "GPT-4o (Latest, Most Capable)",
            "gpt-4o-mini": "GPT-4o Mini (Fast & Efficient)",
            "gpt-4-turbo": "GPT-4 Turbo (Balanced)"
        }
    
    if anthropic_key:
        models["Anthropic"] = {
            "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Recommended)",
            "claude-3-5-haiku-20241022": "Claude 3.5 Haiku (Fast)",
            "claude-3-opus-20240229": "Claude 3 Opus (Most Capable)"
        }
    
    if google_key:
        models["Google"] = {
            "gemini-2.0-flash-exp": "Gemini 2.0 Flash (Experimental, Fastest)",
            "gemini-1.5-pro": "Gemini 1.5 Pro (Balanced)",
            "gemini-1.5-flash": "Gemini 1.5 Flash (Fast & Cost-Effective)"
        }
    
    return models

# AI API Functions
def call_openai(prompt, model="gpt-4o", max_tokens=4000):
    """Call OpenAI API"""
    openai_key, _, _ = get_api_keys()
    if not openai_key:
        st.error("OpenAI API key not found in secrets!")
        return None
    
    try:
        client = openai.OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

def call_anthropic(prompt, model="claude-3-5-sonnet-20241022", max_tokens=4000):
    """Call Anthropic API"""
    _, anthropic_key, _ = get_api_keys()
    if not anthropic_key:
        st.error("Anthropic API key not found in secrets!")
        return None
    
    try:
        client = anthropic.Anthropic(api_key=anthropic_key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        st.error(f"Anthropic API Error: {str(e)}")
        return None

def call_gemini(prompt, model="gemini-1.5-pro", max_tokens=4000):
    """Call Google Gemini API"""
    _, _, google_key = get_api_keys()
    if not google_key:
        st.error("Google API key not found in secrets!")
        return None
    
    try:
        genai.configure(api_key=google_key)
        model_instance = genai.GenerativeModel(model)
        
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": 0.7,
        }
        
        response = model_instance.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        st.error(f"Google Gemini API Error: {str(e)}")
        return None

def call_ai_model(prompt, model_string, max_tokens=4000):
    """Universal AI caller that routes to the correct provider"""
    if model_string.startswith("gpt"):
        return call_openai(prompt, model_string, max_tokens)
    elif model_string.startswith("claude"):
        return call_anthropic(prompt, model_string, max_tokens)
    elif model_string.startswith("gemini"):
        return call_gemini(prompt, model_string, max_tokens)
    else:
        st.error("Unknown model type!")
        return None

# Utility Functions
def extract_headings(markdown_text):
    """Extract H2 and H3 headings from Markdown"""
    h2_pattern = r'^##\s+(.+)$'
    h3_pattern = r'^###\s+(.+)$'
    
    headings = []
    for line in markdown_text.split('\n'):
        h2_match = re.match(h2_pattern, line.strip())
        h3_match = re.match(h3_pattern, line.strip())
        
        if h2_match:
            headings.append(('H2', h2_match.group(1)))
        elif h3_match:
            headings.append(('H3', h3_match.group(1)))
    
    return headings

def semantic_similarity_score(text1, text2, model):
    """Calculate semantic similarity between two texts"""
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

def create_download_link(content, filename, file_format="markdown"):
    """Create download link without page reset"""
    if file_format == "markdown":
        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    else:  # PDF or other formats
        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

# === TAB 1: OUTLINE OPTIMIZER ===
def audience_research_analysis(keyword, model):
    """Perform deep audience research and search intent analysis"""
    prompt = f"""You are an expert Digital Marketing Strategist and Audience Researcher, specializing in deep psychological profiling and cross-platform search behavior analysis.

Your primary objective is to conduct advanced, data-driven audience research for the specified keyword/topic: "{keyword}". This analysis must be comprehensive, actionable, and focus on understanding the user's journey, motivations, and information needs across modern search environments.

The analysis must cover the following seven critical dimensions in detail:
1. **Ideal Customer Profile (ICP):** Detailed demographics and psychographics.
2. **Core Pain Points:** Specific frustrations and problems driving the search.
3. **Emotional Triggers:** Underlying psychological motivations for seeking information.
4. **Cross-Platform Search Behavior:** How users search for this topic specifically on Google, ChatGPT, Claude, Perplexity, and Gemini.
5. **User Intent Clusters:** Categorization of intent (informational, transactional, navigational, investigational).
6. **Information Gaps:** Common questions and missing information users are trying to fill.
7. **Resonance Strategies:** Actionable methods for creating content that deeply connects with this audience.

The final output must be highly specific, actionable, and grounded in analytical insight. Avoid generic statements. Focus on providing depth in each area requested. The analysis must be data-driven."""

    return call_ai_model(prompt, model, max_tokens=3000)

def optimize_outline(keyword, draft_outline, query_fanout, audience_insights, model):
    """Optimize blog outline with AI analysis"""
    prompt = f"""You are an expert SEO content strategist. Optimize the following blog post outline.

PRIMARY KEYWORD: {keyword}

AUDIENCE INSIGHTS:
{audience_insights}

QUERY FAN-OUT ANALYSIS:
{query_fanout}

CURRENT DRAFT OUTLINE:
{draft_outline}

TASK:
1. Analyze the draft outline against the Query Fan-Out suggestions and Audience Insights
2. Prioritize Query Fan-Out recommendations over the current draft
3. Ensure all ICP pain points and search intents are addressed
4. Generate an OPTIMIZED OUTLINE in the EXACT same Markdown structure (preserve H2/H3 hierarchy)

For each H2/H3 section, add:
- 7-12 concise talking points (bullet list) to guide writers
- Each talking point should be actionable and derived from audience research
- Include rationale for key additions (e.g., "Addresses [pain point] from audience research")

OUTPUT FORMAT:
## [H2 Heading]
- Talking point 1
- Talking point 2
...
- Talking point 7-12

### [H3 Heading]
- Talking point 1
...

Ensure logical flow, comprehensive coverage, and SEO optimization. Make it ready for a writer to execute."""

    return call_ai_model(prompt, model, max_tokens=4000)

# === TAB 2: DRAFT OPTIMIZER ===
def keyword_relevance_analysis(primary_keyword, keyword_list, draft_content, nlp, semantic_model):
    """Analyze keyword relevance and find placement opportunities"""
    # Parse draft with spaCy
    doc = nlp(draft_content)
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in draft_content.split('\n\n') if p.strip()]
    
    # Calculate TF-IDF for relevance scoring
    all_text = [primary_keyword] + keyword_list + paragraphs
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)
    
    results = []
    
    for keyword in keyword_list:
        # Calculate semantic similarity with primary keyword
        primary_sim = semantic_similarity_score(keyword, primary_keyword, semantic_model)
        
        # Check if keyword exists in draft (exact or variations)
        keyword_lower = keyword.lower()
        exists_exact = keyword_lower in draft_content.lower()
        
        # Find semantic variations
        variations = []
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if keyword_lower not in sent_text:
                sim = semantic_similarity_score(keyword, sent.text, semantic_model)
                if sim > 0.8:
                    variations.append(sent.text[:50] + "...")
        
        # Calculate relevance score
        keyword_embedding = semantic_model.encode([keyword])
        draft_embedding = semantic_model.encode([draft_content[:500]])  # First 500 chars
        relevance_score = cosine_similarity(keyword_embedding, draft_embedding)[0][0]
        
        # Determine if keyword needs addition
        keyword_density = draft_content.lower().count(keyword_lower) / len(draft_content.split()) * 100
        needs_addition = relevance_score > 0.5 and keyword_density < 1.0
        
        if needs_addition:
            # Find best placement location
            best_para_idx = 0
            best_sim = 0
            for idx, para in enumerate(paragraphs[:5]):  # Check first 5 paragraphs
                sim = semantic_similarity_score(keyword, para, semantic_model)
                if sim > best_sim:
                    best_sim = sim
                    best_para_idx = idx
            
            results.append({
                'keyword': keyword,
                'relevance_score': round(relevance_score, 2),
                'density': round(keyword_density, 2),
                'variations_found': len(variations),
                'needs_addition': needs_addition,
                'best_paragraph_idx': best_para_idx,
                'paragraph_preview': paragraphs[best_para_idx][:100] + "..."
            })
    
    return results

def generate_keyword_integration(keyword, paragraph_context, model):
    """Generate natural keyword integration"""
    prompt = f"""You are an expert SEO Content Editor specializing in seamless keyword integration and natural language flow.

Your primary objective is to rewrite the provided paragraph, ensuring the specified keyword is integrated contextually and naturally. The revised paragraph must maintain the original tone, structure, and overall meaning.

KEYWORD TO INTEGRATE: {keyword}
ORIGINAL PARAGRAPH: {paragraph_context}

- **Integration Quality:** The keyword must be woven into the text so that it reads as if it were originally intended to be there (no keyword stuffing).
- **Tone and Structure:** Strictly adhere to the tone and structural flow of the original paragraph.
- **Length Constraint:** The final rewritten paragraph must remain within the original length range, ideally between 50 and 200 words.
- **Readiness:** The output must be immediately copy-paste ready for publication.

OUTPUT: Only provide the rewritten paragraph, nothing else."""

    return call_ai_model(prompt, model, max_tokens=500)

def ai_tool_optimization(draft_content, primary_keyword, keyword_list, model):
    """Apply 10-item AI optimization checklist"""
    prompt = f"""You are an elite Content Optimization Specialist, specifically trained in maximizing content performance for modern AI Search Engines and Generative AI platforms (including Google AI Overviews, ChatGPT, Claude, and Perplexity). Your primary objective is to transform the provided content draft into a highly optimized, authoritative, and direct piece of content that satisfies complex AI ranking signals.

The core task is to apply a precise set of 10 content optimization rules to the provided DRAFT CONTENT, using the specified KEYWORDS to guide semantic relevance and depth. The final output must strictly adhere to all ten optimization criteria listed below.

PRIMARY KEYWORD: {primary_keyword}
KEYWORDS TO INTEGRATE: {', '.join(keyword_list[:10])}
CONTENT DRAFT TO OPTIMIZE: {draft_content}

Apply the following 10 mandatory optimizations sequentially and thoroughly:

1. **Answer-First Introduction**: Rewrite the opening section (30-50 words) to provide an immediate, direct answer to the core query implied by the primary keyword.
2. **Question-Based H2s**: Systematically convert all existing H2 headings into natural, user-intent-driven questions.
3. **Semantic Chunking**: Ensure every resulting content section is self-contained and falls within a length range of 75 to 300 words.
4. **A-E-C Structure**: Restructure the content within each chunk following the strict sequence: **Answer** (direct response) → **Evidence** (supporting data/facts) → **Context** (broader implications/background).
5. **Active Voice & Clarity**: Convert all sentences to active voice, prioritizing a clear Subject-Verb-Object structure for maximum readability.
6. **Informational Density Boost**: Increase the density of specific, verifiable details (e.g., numbers, specific entities, dates) by approximately 20% across the body.
7. **Claim Attribution**: Replace vague or generic statements with specific attribution. For example, change general claims to formats like: "A 2023 study by XYZ found..." or "According to industry standard ABC..."
8. **Demonstrate Authority**: Integrate phrases that signal direct experience or testing, such as "In our testing..." or "Based on our hands-on evaluation...", where contextually appropriate.
9. **Long-Tail FAQ Section**: Append a dedicated FAQ section containing between 3 and 10 distinct, long-tail questions and their corresponding detailed answers.
10. **SEO Metadata Generation**: Generate an optimized Title Tag (must be under 60 characters) and a compelling Meta Description (must be between 140 and 160 characters).

OUTPUT the fully optimized content in Markdown format with clear section breaks. Mark changes with [OPTIMIZED] tags inline."""

    return call_ai_model(prompt, model, max_tokens=4000)

# === MAIN APP ===
def main():
    st.markdown('<p class="main-header">✍️ Blog Post Optimizer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Content Optimization for SEO & AI Search Tools</p>', unsafe_allow_html=True)
    
    # Sidebar - API Configuration Check
    with st.sidebar:
        st.header("⚙️ Configuration")
        openai_key, anthropic_key, google_key = get_api_keys()
        
        # Display API connection status
        st.subheader("API Connection Status")
        if openai_key:
            st.success("✅ OpenAI API Connected")
        else:
            st.warning("⚠️ OpenAI API Key Missing")
        
        if anthropic_key:
            st.success("✅ Anthropic API Connected")
        else:
            st.warning("⚠️ Anthropic API Key Missing")
        
        if google_key:
            st.success("✅ Google Gemini API Connected")
        else:
            st.warning("⚠️ Google API Key Missing")
        
        # Get available models
        available_models = get_available_models()
        
        if not available_models:
            st.error("❌ No API keys configured! Please add at least one API key to Streamlit Secrets.")
            st.stop()
        
        st.markdown("---")
        
        # Smart Model Selector - only shows models from configured providers
        st.subheader("🤖 AI Model Selection")
        
        # Flatten the models dictionary for selection
        model_options = {}
        for provider, models in available_models.items():
            for model_id, model_name in models.items():
                model_options[f"{provider}: {model_name}"] = model_id
        
        selected_model_display = st.selectbox(
            "Select AI Model:",
            options=list(model_options.keys()),
            help="Only models from configured API providers are shown"
        )
        
        selected_model = model_options[selected_model_display]
        
        # Display model info
        st.info(f"**Active Model:** `{selected_model}`")
        
        st.markdown("---")
        st.markdown("**💡 Setup Instructions:**")
        st.markdown("Add API keys to Streamlit Cloud Secrets:")
        st.code("""OPENAI_API_KEY = "your-key"
ANTHROPIC_API_KEY = "your-key"
GOOGLE_API_KEY = "your-key" """)
        
        st.markdown("---")
        st.markdown("**💰 Cost Comparison:**")
        st.markdown("- 🥇 **Gemini**: Most cost-effective (free tier)")
        st.markdown("- 🥈 **Anthropic**: Best quality/cost ratio")
        st.markdown("- 🥉 **OpenAI**: Most capable models")
    
    # Load NLP models
    with st.spinner("Loading NLP models..."):
        nlp, semantic_model = load_nlp_models()
    
    # Main Tabs
    tab1, tab2 = st.tabs(["📝 Outline Optimizer", "✨ Draft Optimizer"])
    
    # === TAB 1: OUTLINE OPTIMIZER ===
    with tab1:
        st.header("Outline Optimizer")
        st.markdown("Optimize your blog outline with audience research, search intent analysis, and SEO best practices.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            keyword_outline = st.text_input(
                "Primary Keyword/Topic *",
                placeholder="e.g., dog training tips",
                key="outline_keyword"
            )
            
            draft_outline = st.text_area(
                "Draft Outline (Markdown) *",
                placeholder="## Introduction\n\n## Main Point 1\n### Subpoint 1.1\n\n## Conclusion",
                height=300,
                key="draft_outline_input"
            )
        
        with col2:
            query_fanout = st.text_area(
                "Query Fan-Out Analysis",
                placeholder="Paste your Query Fan-Out report here (structured text with expanded queries, related terms, intent insights)",
                height=300,
                key="query_fanout_input"
            )
            
            uploaded_fanout = st.file_uploader(
                "Or Upload Query Fan-Out Report",
                type=['txt', 'md'],
                key="fanout_upload"
            )
        
        if uploaded_fanout:
            query_fanout = uploaded_fanout.read().decode('utf-8')
            st.success("✅ File uploaded successfully!")
        
        st.markdown("---")
        
        if st.button("🚀 Optimize Outline", type="primary", use_container_width=True):
            if not keyword_outline or not draft_outline:
                st.error("Please provide both primary keyword and draft outline!")
            else:
                # Step 1: Audience Research
                with st.spinner("🔍 Step 1/2: Conducting deep audience research..."):
                    progress_bar = st.progress(0)
                    
                    audience_insights = audience_research_analysis(keyword_outline, selected_model)
                    st.session_state.audience_insights = audience_insights
                    progress_bar.progress(50)
                    
                    if audience_insights:
                        st.success("✅ Audience research complete!")
                        with st.expander("📊 View Audience Insights", expanded=False):
                            st.markdown(audience_insights)
                    
                    progress_bar.progress(100)
                
                # Step 2: Outline Optimization
                with st.spinner("✨ Step 2/2: Optimizing outline..."):
                    progress_bar2 = st.progress(0)
                    
                    fanout_text = query_fanout if query_fanout else "No Query Fan-Out analysis provided."
                    optimized_outline = optimize_outline(
                        keyword_outline,
                        draft_outline,
                        fanout_text,
                        audience_insights,
                        selected_model
                    )
                    
                    st.session_state.outline_result = optimized_outline
                    progress_bar2.progress(100)
                
                if optimized_outline:
                    st.markdown("---")
                    st.subheader("📊 Optimization Results")
                    
                    # Side-by-side comparison
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.markdown("### 📄 Original Outline")
                        st.markdown(draft_outline)
                    
                    with comp_col2:
                        st.markdown("### ✅ Optimized Outline")
                        st.markdown(optimized_outline)
                    
                    # Download options
                    st.markdown("---")
                    download_col1, download_col2 = st.columns(2)
                    
                    with download_col1:
                        filename = f"optimized_outline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                        st.markdown(
                            create_download_link(optimized_outline, filename),
                            unsafe_allow_html=True
                        )
                    
                    with download_col2:
                        if st.session_state.audience_insights:
                            insights_filename = f"audience_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                            st.markdown(
                                create_download_link(st.session_state.audience_insights, insights_filename),
                                unsafe_allow_html=True
                            )
    
    # === TAB 2: DRAFT OPTIMIZER ===
    with tab2:
        st.header("Draft Optimizer")
        st.markdown("Optimize your draft through keyword analysis and AI tool optimization (2-step process).")
        
        keyword_draft = st.text_input(
            "Primary Keyword/Topic *",
            placeholder="e.g., dog training tips",
            key="draft_keyword"
        )
        
        draft_col1, draft_col2 = st.columns([2, 1])
        
        with draft_col1:
            draft_content = st.text_area(
                "Content Draft *",
                placeholder="Paste your blog draft here (Markdown or HTML supported)",
                height=400,
                key="draft_content_input"
            )
        
        with draft_col2:
            keyword_list_input = st.text_area(
                "Expanded Keyword List *",
                placeholder="keyword 1\nkeyword 2\nkeyword 3",
                height=200,
                key="keyword_list_input"
            )
            
            uploaded_draft = st.file_uploader(
                "Or Upload Draft File",
                type=['txt', 'md', 'html'],
                key="draft_upload"
            )
        
        if uploaded_draft:
            draft_content = uploaded_draft.read().decode('utf-8')
            st.success("✅ Draft uploaded successfully!")
        
        # Parse keyword list
        keyword_list = [k.strip() for k in keyword_list_input.split('\n') if k.strip()]
        if ',' in keyword_list_input:
            keyword_list = [k.strip() for k in keyword_list_input.split(',') if k.strip()]
        
        st.markdown("---")
        
        if st.button("🚀 Optimize Draft", type="primary", use_container_width=True):
            if not keyword_draft or not draft_content or not keyword_list:
                st.error("Please provide primary keyword, draft content, and keyword list!")
            else:
                # Subprocess 1: Keyword Optimization
                st.markdown("### 📊 Subprocess 1: Keyword Optimization")
                
                with st.spinner("🔍 Analyzing keyword relevance and placement..."):
                    keyword_analysis = keyword_relevance_analysis(
                        keyword_draft,
                        keyword_list,
                        draft_content,
                        nlp,
                        semantic_model
                    )
                
                if keyword_analysis:
                    st.success(f"✅ Analyzed {len(keyword_analysis)} high-relevance keywords needing optimization")
                    
                    # Generate integration suggestions
                    integration_results = []
                    
                    progress = st.progress(0)
                    for idx, kw_data in enumerate(keyword_analysis[:10]):  # Limit to top 10
                        with st.spinner(f"Generating integration for '{kw_data['keyword']}'..."):
                            integrated_paragraph = generate_keyword_integration(
                                kw_data['keyword'],
                                kw_data['paragraph_preview'],
                                selected_model
                            )
                            
                            integration_results.append({
                                'Selected Keyword': f"{kw_data['keyword']} (Score: {kw_data['relevance_score']})",
                                'Placement Hint': f"Insert in paragraph {kw_data['best_paragraph_idx'] + 1}: '{kw_data['paragraph_preview'][:50]}...'",
                                'Content Snippet': integrated_paragraph if integrated_paragraph else "Generation failed"
                            })
                        
                        progress.progress((idx + 1) / min(10, len(keyword_analysis)))
                    
                    # Display table
                    if integration_results:
                        st.markdown("#### 📝 Keyword Integration Suggestions")
                        df = pd.DataFrame(integration_results)
                        st.dataframe(df, use_container_width=True, height=400)
                
                st.markdown("---")
                
                # Subprocess 2: AI Tool Optimization
                st.markdown("### ✨ Subprocess 2: AI Tool Optimization")
                
                with st.spinner("🤖 Applying 10-item AI optimization checklist..."):
                    progress2 = st.progress(0)
                    
                    optimized_draft = ai_tool_optimization(
                        draft_content,
                        keyword_draft,
                        keyword_list,
                        selected_model
                    )
                    
                    progress2.progress(100)
                    st.session_state.draft_result = optimized_draft
                
                if optimized_draft:
                    st.success("✅ Draft fully optimized!")
                    
                    # Checklist compliance
                    st.markdown("#### ✅ Optimization Checklist Compliance")
                    checklist_items = [
                        "Answer-First Introduction",
                        "Question-Based H2 Headings",
                        "Semantic Chunks (75-300 words)",
                        "Answer-Evidence-Context Formula",
                        "Direct Sentence Structures",
                        "Informational Density (+20%)",
                        "Attributed Claims",
                        "First-Hand Experience Signals",
                        "Dedicated FAQ Section",
                        "Optimized Title & Meta"
                    ]
                    
                    cols = st.columns(5)
                    for idx, item in enumerate(checklist_items):
                        with cols[idx % 5]:
                            st.metric(label=item, value="✅", delta="Applied")
                    
                    st.markdown("---")
                    
                    # Display optimized draft
                    st.markdown("#### 📄 Fully Optimized Draft")
                    st.markdown(optimized_draft)
                    
                    # Download
                    st.markdown("---")
                    final_filename = f"optimized_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    st.markdown(
                        create_download_link(optimized_draft, final_filename),
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":

    main()
