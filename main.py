import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from typing import List, Dict, Any
import io
import re
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# For PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Silicon Sampling Simulation Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DeepSeekAPI:
    """DeepSeek API client for cognitive layer processing"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_relationships(self, data: pd.DataFrame, dependent_var: str, 
                            independent_vars: List[str], control_vars: List[str],
                            variable_descriptions: Dict[str, str], context: str = "") -> Dict[str, Any]:
        """Analyze relationships between variables using DeepSeek API"""
        
        # Prepare data summary for API
        data_summary = {
            "dependent_variable": dependent_var,
            "independent_variables": independent_vars,
            "control_variables": control_vars,
            "data_shape": data.shape,
            "variable_descriptions": variable_descriptions,
            "context": context,
            "sample_statistics": {}
        }
        
        # Add statistical summaries for each variable
        for var in [dependent_var] + independent_vars + control_vars:
            if var in data.columns:
                if data[var].dtype in ['int64', 'float64']:
                    data_summary["sample_statistics"][var] = {
                        "mean": float(data[var].mean()),
                        "std": float(data[var].std()),
                        "min": float(data[var].min()),
                        "max": float(data[var].max()),
                        "median": float(data[var].median())
                    }
                else:
                    data_summary["sample_statistics"][var] = {
                        "unique_values": data[var].nunique(),
                        "most_common": data[var].mode().iloc[0] if not data[var].mode().empty else "N/A"
                    }
        
        prompt = f"""
        As an expert data scientist, provide a simple, actionable analysis of this experimental data:

        Data Summary:
        {json.dumps(data_summary, indent=2)}

        Variable Descriptions:
        {json.dumps(variable_descriptions, indent=2)}

        Additional Context:
        {context}

        Please provide a CLEAR, USER-FRIENDLY analysis with these sections:

        ## 🔍 KEY FINDINGS (What Your Data Shows)
        - Summarize the main relationships in plain English
        - Focus on practical insights for decision-making

        ## 📊 SIMPLE REGRESSION MODEL
        **Equation**: {dependent_var} = [provide simple equation with actual numbers]
        
        **What this means**: Explain in plain language what happens when variables change
        
        **Strength**: Rate the model strength (Weak/Moderate/Strong) and expected R²

        ## ⚙️ CONTROL FACTORS
        How {str(control_vars)} affect your results (keep it simple)

        ## 💡 RECOMMENDED ADDITIONAL VARIABLES (for future studies)
        List 3-5 specific variables that would improve your research:
        - Variable Name: Why it matters for your research

        ## ⚠️ IMPORTANT LIMITATIONS
        What to be careful about when interpreting these results

        ## 🎯 ACTIONABLE RECOMMENDATIONS
        What should you do next based on these findings?

        Keep explanations simple and focus on practical value for the researcher.
        """
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 3000
                }
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling DeepSeek API: {str(e)}"
    
    def generate_synthetic_data(self, original_data: pd.DataFrame, 
                              dependent_var: str, independent_vars: List[str],
                              control_vars: List[str], additional_vars: List[str],
                              n_samples: int = 100) -> pd.DataFrame:
        """Generate actual synthetic data based on learned relationships"""
        
        # Get basic statistics from original data
        stats = {}
        all_vars = [dependent_var] + independent_vars + control_vars
        
        for var in all_vars:
            if var in original_data.columns:
                if original_data[var].dtype in ['int64', 'float64']:
                    stats[var] = {
                        'mean': float(original_data[var].mean()),
                        'std': float(original_data[var].std()),
                        'min': float(original_data[var].min()),
                        'max': float(original_data[var].max()),
                        'type': 'numeric'
                    }
                else:
                    stats[var] = {
                        'unique_values': original_data[var].unique().tolist(),
                        'type': 'categorical'
                    }
        
        prompt = f"""
        Generate realistic synthetic data parameters for the following variables:
        
        Original data statistics: {json.dumps(stats, indent=2)}
        
        Dependent variable: {dependent_var}
        Independent variables: {independent_vars}
        Control variables: {control_vars}  
        Additional variables to include: {additional_vars}
        
        Provide ONLY a JSON response with the following structure:
        {{
            "coefficients": {{
                "intercept": <number>,
                "{independent_vars[0] if independent_vars else 'var1'}": <number>,
                ...
            }},
            "error_variance": <number>,
            "variable_ranges": {{
                "var_name": {{"min": <number>, "max": <number>, "distribution": "normal/uniform"}},
                ...
            }},
            "additional_variables": {{
                "var_name": {{"description": "...", "relationship": "correlation with...", "range": {{"min": <number>, "max": <number>}}}},
                ...
            }}
        }}
        
        Make coefficients realistic based on the data patterns. Do not include any explanation text.
        """
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1500
                }
            )
            
            if response.status_code == 200:
                ai_response = response.json()['choices'][0]['message']['content']
                
                # Try to extract JSON from the response
                try:
                    # Clean the response to extract JSON
                    json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                    if json_match:
                        params = json.loads(json_match.group())
                        return self._create_synthetic_dataset(original_data, params, all_vars + additional_vars, n_samples)
                    else:
                        return self._create_fallback_synthetic_data(original_data, all_vars, n_samples)
                        
                except json.JSONDecodeError:
                    return self._create_fallback_synthetic_data(original_data, all_vars, n_samples)
                    
            else:
                return self._create_fallback_synthetic_data(original_data, all_vars, n_samples)
                
        except Exception as e:
            return self._create_fallback_synthetic_data(original_data, all_vars, n_samples)
    
    def _create_synthetic_dataset(self, original_data: pd.DataFrame, params: Dict, 
                                variables: List[str], n_samples: int) -> pd.DataFrame:
        """Create synthetic dataset using intelligent variable type detection"""
        
        synthetic_data = {}
        
        for var in variables:
            if var in original_data.columns:
                # Detect variable characteristics
                var_info = detect_variable_type(original_data[var])
                
                # Generate synthetic data matching exact characteristics
                synthetic_values = generate_synthetic_variable(var_info, n_samples, original_data[var])
                synthetic_data[var] = synthetic_values
            else:
                # Handle additional variables not in original data
                # Generate reasonable defaults based on variable name
                if any(keyword in var.lower() for keyword in ['age']):
                    synthetic_data[var] = np.random.randint(18, 80, n_samples)
                elif any(keyword in var.lower() for keyword in ['income', 'salary']):
                    synthetic_data[var] = np.random.normal(50000, 20000, n_samples)
                elif any(keyword in var.lower() for keyword in ['satisfaction', 'rating', 'score']):
                    synthetic_data[var] = np.random.randint(1, 6, n_samples)  # 1-5 scale
                else:
                    # Default to normal distribution
                    synthetic_data[var] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(synthetic_data)
    
    def _create_fallback_synthetic_data(self, original_data: pd.DataFrame, 
                                      variables: List[str], n_samples: int) -> pd.DataFrame:
        """Fallback method using intelligent variable type detection"""
        
        synthetic_data = {}
        
        for var in variables:
            if var in original_data.columns:
                # Use intelligent variable detection
                var_info = detect_variable_type(original_data[var])
                synthetic_values = generate_synthetic_variable(var_info, n_samples, original_data[var])
                synthetic_data[var] = synthetic_values
            else:
                # Handle additional variables with smart defaults
                if any(keyword in var.lower() for keyword in ['age']):
                    synthetic_data[var] = np.random.randint(18, 80, n_samples)
                elif any(keyword in var.lower() for keyword in ['income', 'salary']):
                    synthetic_data[var] = np.random.normal(50000, 20000, n_samples)
                elif any(keyword in var.lower() for keyword in ['satisfaction', 'rating', 'score']):
                    synthetic_data[var] = np.random.randint(1, 6, n_samples)
                else:
                    synthetic_data[var] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(synthetic_data)

    def analyze_pdf_document(self, pdf_content: bytes) -> Dict[str, Any]:
        """Analyze PDF document for variable context and suggestions with smart experimental design detection"""
        
        if not PDF_AVAILABLE:
            return {"analysis": "PDF analysis not available. Please install PyPDF2.", "variable_suggestions": {}, "experimental_design": {}}
        
        try:
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Truncate text if too long (API limits)
            if len(text) > 12000:
                text = text[:12000] + "..."
            
            prompt = f"""
            You are an expert in experimental design and survey methodology. Analyze this research document and extract detailed variable information with special focus on experimental conditions, randomization, and treatments.

            Document text:
            {text}

            Please provide a comprehensive JSON response with this exact structure:

            {{
                "analysis": "Brief summary of the research context, methodology, and experimental design",
                "experimental_design": {{
                    "type": "experiment/survey/observational",
                    "treatments": ["list of experimental conditions/treatments"],
                    "randomization": "description of randomization method",
                    "dependent_variables": ["list of outcome variables"],
                    "independent_variables": ["list of predictor/treatment variables"],
                    "control_variables": ["list of control/demographic variables"]
                }},
                "variable_suggestions": {{
                    "exact_variable_name": "Clear description of what this measures (include scale if mentioned)",
                    "condition": "Experimental condition/treatment assignment (e.g., control=0, treatment=1)",
                    "randomization": "Random assignment variable or grouping variable",
                    "group": "Group assignment or condition indicator",
                    "treatment": "Treatment condition variable",
                    "manipulation": "Experimental manipulation variable"
                }},
                "scale_information": {{
                    "variable_name": {{
                        "type": "likert/binary/categorical/continuous",
                        "range": "e.g., 1-7 scale",
                        "labels": ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]
                    }}
                }},
                "additional_variables": {{
                    "suggested_var_1": "Why this variable would enhance the study based on the research context",
                    "suggested_var_2": "Why this variable would enhance the study based on the research context"
                }}
            }}

            IMPORTANT INSTRUCTIONS:
            1. Look for experimental conditions, treatments, groups, manipulations
            2. Identify randomization procedures, assignment methods
            3. Extract exact variable names as they appear in the document
            4. Pay special attention to condition variables, group assignments
            5. Look for scale information (1-5, 1-7, Likert scales, etc.)
            6. Identify survey questions and their response scales
            7. Find demographic variables and control measures
            8. Extract any coding schemes (0=control, 1=treatment, etc.)

            Focus on experimental design elements and be very specific about variable names and scales.
            """
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 3500
                }
            )
            
            if response.status_code == 200:
                ai_response = response.json()['choices'][0]['message']['content']
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON in the response
                    json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                    if json_match:
                        parsed_result = json.loads(json_match.group())
                        # Ensure all required keys exist
                        default_structure = {
                            "analysis": "Analysis completed",
                            "experimental_design": {},
                            "variable_suggestions": {},
                            "scale_information": {},
                            "additional_variables": {}
                        }
                        for key in default_structure:
                            if key not in parsed_result:
                                parsed_result[key] = default_structure[key]
                        return parsed_result
                    else:
                        return {
                            "analysis": ai_response, 
                            "experimental_design": {},
                            "variable_suggestions": {},
                            "scale_information": {},
                            "additional_variables": {}
                        }
                except json.JSONDecodeError as e:
                    # Fallback: try to extract key information manually
                    return self._manual_extraction_fallback(ai_response, text)
            else:
                return {
                    "analysis": f"Error analyzing PDF: {response.status_code}", 
                    "experimental_design": {},
                    "variable_suggestions": {},
                    "scale_information": {},
                    "additional_variables": {}
                }
                
        except Exception as e:
            return {
                "analysis": f"Error processing PDF: {str(e)}", 
                "experimental_design": {},
                "variable_suggestions": {},
                "scale_information": {},
                "additional_variables": {}
            }
    
    def _manual_extraction_fallback(self, ai_response: str, original_text: str) -> Dict[str, Any]:
        """Fallback method to extract key information when JSON parsing fails"""
        
        result = {
            "analysis": ai_response,
            "experimental_design": {},
            "variable_suggestions": {},
            "scale_information": {},
            "additional_variables": {}
        }
        
        # Simple keyword-based extraction from original text
        text_lower = original_text.lower()
        
        # Look for experimental design keywords
        if any(word in text_lower for word in ['experiment', 'treatment', 'condition', 'randomiz', 'control group']):
            result["experimental_design"]["type"] = "experiment"
        elif any(word in text_lower for word in ['survey', 'questionnaire', 'interview']):
            result["experimental_design"]["type"] = "survey"
        
        # Look for common variable patterns
        variable_suggestions = {}
        
        # Common experimental variables
        if 'condition' in text_lower:
            variable_suggestions["condition"] = "Experimental condition or treatment assignment"
        if 'group' in text_lower:
            variable_suggestions["group"] = "Group assignment variable"
        if 'treatment' in text_lower:
            variable_suggestions["treatment"] = "Treatment condition indicator"
        if any(word in text_lower for word in ['randomiz', 'random assign']):
            variable_suggestions["randomization"] = "Random assignment indicator"
        
        # Look for scale patterns
        scale_patterns = re.findall(r'(\d+[-‒–—]\d+|\d+\s*point)', text_lower)
        if scale_patterns:
            result["scale_information"]["detected_scales"] = {
                "patterns": scale_patterns,
                "description": "Detected scale patterns in document"
            }
        
        result["variable_suggestions"] = variable_suggestions
        return result

def detect_variable_type(series: pd.Series) -> Dict[str, Any]:
    """Detect variable type and characteristics for exact replication"""
    
    var_info = {
        'type': 'unknown',
        'min_val': None,
        'max_val': None,
        'unique_count': series.nunique(),
        'is_integer': False,
        'scale_type': 'unknown'
    }
    
    # Remove NaN values for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return var_info
    
    # Check if numeric
    if pd.api.types.is_numeric_dtype(clean_series):
        var_info['type'] = 'numeric'
        var_info['min_val'] = float(clean_series.min())
        var_info['max_val'] = float(clean_series.max())
        var_info['is_integer'] = clean_series.dtype in ['int32', 'int64'] or all(clean_series == clean_series.astype(int))
        
        # Detect scale patterns
        unique_vals = sorted(clean_series.unique())
        
        # Binary variable detection
        if len(unique_vals) == 2:
            var_info['scale_type'] = 'binary'
            var_info['categories'] = unique_vals
        
        # Likert scale detection (1-5, 1-7, etc.)
        elif (len(unique_vals) <= 10 and 
              var_info['min_val'] >= 1 and 
              var_info['is_integer'] and
              unique_vals == list(range(int(var_info['min_val']), int(var_info['max_val']) + 1))):
            var_info['scale_type'] = f"likert_{int(var_info['min_val'])}_{int(var_info['max_val'])}"
            var_info['categories'] = unique_vals
        
        # Continuous numeric
        else:
            var_info['scale_type'] = 'continuous'
    
    # Categorical variable
    else:
        var_info['type'] = 'categorical'
        var_info['scale_type'] = 'categorical'
        var_info['categories'] = clean_series.unique().tolist()
        var_info['category_counts'] = clean_series.value_counts().to_dict()
    
    return var_info

def generate_synthetic_variable(var_info: Dict, n_samples: int, original_series: pd.Series) -> np.ndarray:
    """Generate synthetic data that exactly matches original variable characteristics"""
    
    if var_info['type'] == 'numeric':
        
        if var_info['scale_type'] == 'binary':
            # Binary variable - maintain exact proportions
            unique_vals = var_info['categories']
            value_counts = original_series.value_counts(normalize=True)
            return np.random.choice(unique_vals, n_samples, p=[value_counts.get(val, 0.5) for val in unique_vals])
        
        elif var_info['scale_type'].startswith('likert'):
            # Likert scale - maintain distribution
            unique_vals = var_info['categories']
            value_counts = original_series.value_counts(normalize=True)
            probs = [value_counts.get(val, 1/len(unique_vals)) for val in unique_vals]
            return np.random.choice(unique_vals, n_samples, p=probs)
        
        else:
            # Continuous - normal distribution with exact bounds
            mean = float(original_series.mean())
            std = float(original_series.std())
            
            # Generate values and clip to exact bounds
            synthetic_vals = np.random.normal(mean, std, n_samples)
            synthetic_vals = np.clip(synthetic_vals, var_info['min_val'], var_info['max_val'])
            
            # If original was integer, round to integers
            if var_info['is_integer']:
                synthetic_vals = np.round(synthetic_vals).astype(int)
            
            return synthetic_vals
    
    else:
        # Categorical variable - maintain exact proportions
        categories = var_info['categories']
        value_counts = original_series.value_counts(normalize=True)
        probs = [value_counts.get(cat, 1/len(categories)) for cat in categories]
        return np.random.choice(categories, n_samples, p=probs)
    """Extract suggested variables from AI analysis"""
    variables = []
    
    # Look for suggested variables section
    lines = analysis_text.split('\n')
    in_variables_section = False
    
    for line in lines:
        # Updated patterns to match new format
        if any(keyword in line.upper() for keyword in ['RECOMMENDED ADDITIONAL', 'SUGGESTED ADDITIONAL', 'ADDITIONAL VARIABLES']):
            in_variables_section = True
            continue
        elif line.startswith('#') and in_variables_section:
            break
        elif in_variables_section and line.strip():
            # Extract variable names from bullet points or dashes
            if line.strip().startswith(('-', '*', '•')):
                # Look for pattern "Variable Name:" or just the first word
                var_match = re.search(r'[-*•]\s*([^:]+)', line)
                if var_match:
                    var_name = var_match.group(1).strip()
                    # Clean up variable names
                    var_name = re.sub(r'\(.*?\)', '', var_name).strip()  # Remove parentheses content
                    if len(var_name) < 50 and len(var_name) > 2:  # Reasonable variable name length
                        variables.append(var_name)
    
    return variables[:8]  # Limit to 8 suggestions

def extract_suggested_variables(analysis_text: str) -> List[str]:
    """Create download button for synthetic data"""
    
    if file_format == 'csv':
        csv_data = data.to_csv(index=False)
        st.download_button(
            label=f"📥 Download as CSV ({len(data)} rows)",
            data=csv_data,
            file_name=f"{filename}.csv",
            mime="text/csv"
        )
    elif file_format == 'json':
        json_data = data.to_json(orient='records', indent=2)
        st.download_button(
            label=f"📥 Download as JSON ({len(data)} rows)",
            data=json_data,
            file_name=f"{filename}.json",
            mime="application/json"
        )

def load_and_preview_data(uploaded_file) -> pd.DataFrame:
    """Load and preview uploaded Excel file"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_correlation_heatmap(df: pd.DataFrame, selected_vars: List[str]):
    """Create correlation heatmap for selected variables"""
    numeric_vars = [var for var in selected_vars if var in df.columns and df[var].dtype in ['int64', 'float64']]
    
    if len(numeric_vars) < 2:
        st.warning("Need at least 2 numeric variables for correlation analysis")
        return None
    
    corr_matrix = df[numeric_vars].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Variable Correlation Matrix"
    )
    
    return fig

def create_relationship_plots(df: pd.DataFrame, dependent_var: str, independent_vars: List[str]):
    """Create scatter plots showing relationships between DV and IVs"""
    
    n_vars = len(independent_vars)
    if n_vars == 0:
        return None
    
    # Create subplots
    cols = min(3, n_vars)
    rows = (n_vars + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{dependent_var} vs {var}" for var in independent_vars],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    for i, var in enumerate(independent_vars):
        row = i // cols + 1
        col = i % cols + 1
        
        if var in df.columns and df[var].dtype in ['int64', 'float64']:
            fig.add_trace(
                go.Scatter(
                    x=df[var],
                    y=df[dependent_var],
                    mode='markers',
                    name=f"{var}",
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(height=300*rows, title_text="Relationship Analysis")
    return fig

def main():
    st.title("🔬 Silicon Sampling Simulation Tool")
    st.markdown("### Virtual Replication and Extension of Experiments")
    
    # Get API key from Streamlit secrets
    try:
        api_key = st.secrets["DeepSeek_API_KEY01"]
        st.sidebar.success("✅ API Key loaded from secrets")
    except Exception as e:
        st.sidebar.error("❌ API Key not found in secrets")
        st.error("Please configure the DeepSeek API key in Streamlit secrets as 'DeepSeek_API_KEY01'")
        st.stop()
    
    # Initialize API client
    deepseek_client = DeepSeekAPI(api_key)
    
    # Main workflow tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Data Upload", "📋 Context & Codebook", "🔍 Variable Selection", "🧠 AI Analysis", "🔮 Simulation"])
    
    with tab1:
        st.header("Data Upload and Preview")
        
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload your experimental data in Excel format"
        )
        
        if uploaded_file is not None:
            df = load_and_preview_data(uploaded_file)
            
            if df is not None:
                st.session_state.df = df
                
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
                
                # Basic statistics
                st.subheader("Basic Statistics")
                st.dataframe(df.describe())
                
                # Data types
                st.subheader("Variable Types")
                type_df = pd.DataFrame({
                    'Variable': df.columns,
                    'Data Type': df.dtypes.astype(str),  # Convert to string to avoid Arrow issues
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(type_df)
    
    with tab2:
        st.header("📋 Context & Codebook Analysis")
        
        # Optional context field
        st.subheader("🔍 Study Context")
        context_input = st.text_area(
            "Provide context about your study (optional but recommended):",
            placeholder="e.g., This is a customer satisfaction survey measuring the impact of service quality on retention rates...",
            height=100,
            key="study_context"
        )
        
        if context_input:
            st.session_state.study_context = context_input
        
        # PDF upload for codebook analysis
        st.subheader("📚 Upload Codebook/Survey PDF")
        
        if PDF_AVAILABLE:
            uploaded_pdf = st.file_uploader(
                "Upload PDF Codebook or Survey Document",
                type=['pdf'],
                help="Upload a PDF document containing variable descriptions, survey questions, or codebook"
            )
            
            if uploaded_pdf is not None:
                pdf_content = uploaded_pdf.read()
                
                if st.button("🔍 Analyze PDF Document"):
                    with st.spinner("Analyzing PDF document..."):
                        pdf_analysis_result = deepseek_client.analyze_pdf_document(pdf_content)
                        st.session_state.pdf_analysis = pdf_analysis_result
                
                # Display PDF analysis results with enhanced experimental design detection
                if 'pdf_analysis' in st.session_state:
                    st.subheader("📄 PDF Analysis Results")
                    
                    pdf_data = st.session_state.pdf_analysis
                    
                    if isinstance(pdf_data, dict):
                        # Display experimental design information
                        if 'experimental_design' in pdf_data and pdf_data['experimental_design']:
                            st.markdown("**🧪 Experimental Design Detected:**")
                            exp_design = pdf_data['experimental_design']
                            
                            design_cols = st.columns(2)
                            with design_cols[0]:
                                if 'type' in exp_design:
                                    st.info(f"**Study Type**: {exp_design['type'].title()}")
                                if 'treatments' in exp_design and exp_design['treatments']:
                                    st.success(f"**Treatments**: {', '.join(exp_design['treatments'])}")
                            
                            with design_cols[1]:
                                if 'randomization' in exp_design and exp_design['randomization']:
                                    st.warning(f"**Randomization**: {exp_design['randomization']}")
                                if 'dependent_variables' in exp_design and exp_design['dependent_variables']:
                                    st.info(f"**Outcome Variables**: {', '.join(exp_design['dependent_variables'])}")
                        
                        # Display analysis summary
                        if 'analysis' in pdf_data:
                            with st.expander("📋 Full Study Context", expanded=False):
                                st.write(pdf_data['analysis'])
                        
                        # Show variable suggestions with scale information
                        if 'variable_suggestions' in pdf_data and pdf_data['variable_suggestions']:
                            st.markdown("**🏷️ Variables Detected (will auto-populate):**")
                            
                            # Group variables by type for better display
                            experimental_vars = {}
                            other_vars = {}
                            
                            for var, desc in pdf_data['variable_suggestions'].items():
                                if any(keyword in var.lower() for keyword in ['condition', 'treatment', 'group', 'randomiz', 'manipulation']):
                                    experimental_vars[var] = desc
                                else:
                                    other_vars[var] = desc
                            
                            if experimental_vars:
                                st.markdown("*Experimental Variables:*")
                                for var, desc in experimental_vars.items():
                                    st.success(f"🧪 **{var}**: {desc}")
                            
                            if other_vars:
                                st.markdown("*Other Variables:*")
                                for var, desc in other_vars.items():
                                    st.info(f"📊 **{var}**: {desc}")
                        
                        # Show scale information if detected
                        if 'scale_information' in pdf_data and pdf_data['scale_information']:
                            st.markdown("**📏 Scale Information Detected:**")
                            scales = pdf_data['scale_information']
                            for var, scale_info in scales.items():
                                if isinstance(scale_info, dict):
                                    scale_text = f"Type: {scale_info.get('type', 'unknown')}"
                                    if 'range' in scale_info:
                                        scale_text += f", Range: {scale_info['range']}"
                                    st.caption(f"• {var}: {scale_text}")
                        
                        # Show additional variable suggestions
                        if 'additional_variables' in pdf_data and pdf_data['additional_variables']:
                            with st.expander("💡 Additional Variables Suggested", expanded=False):
                                for var, reason in pdf_data['additional_variables'].items():
                                    st.write(f"**{var}**: {reason}")
                    else:
                        st.markdown(pdf_data)
        else:
            st.warning("PDF analysis not available. Install PyPDF2 to enable this feature.")
            st.code("pip install PyPDF2", language="bash")
    
    with tab3:
        st.header("Variable Selection and Description")
        
        if 'df' not in st.session_state:
            st.warning("Please upload data first in the Data Upload tab")
            return
        
        df = st.session_state.df
        
        # Variable descriptions with auto-population from PDF
        st.subheader("📝 Variable Descriptions")
        st.markdown("Provide brief descriptions for your variables to help the AI understand the context:")
        
        # Initialize variable descriptions in session state if not present
        if 'var_descriptions' not in st.session_state:
            st.session_state.var_descriptions = {}
        
        # Get PDF suggestions if available with improved matching
        pdf_suggestions = {}
        experimental_info = {}
        scale_info = {}
        
        if 'pdf_analysis' in st.session_state and isinstance(st.session_state.pdf_analysis, dict):
            pdf_data = st.session_state.pdf_analysis
            pdf_suggestions = pdf_data.get('variable_suggestions', {})
            experimental_info = pdf_data.get('experimental_design', {})
            scale_info = pdf_data.get('scale_information', {})
        
        # Enhanced auto-populate from PDF analysis
        if pdf_suggestions:
            st.success(f"✅ Found {len(pdf_suggestions)} variable descriptions from your PDF analysis!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Auto-populate ALL from PDF", type="primary"):
                    matched_count = 0
                    for var_name, description in pdf_suggestions.items():
                        # Enhanced matching logic
                        for col in df.columns:
                            # Exact match (case insensitive)
                            if var_name.lower() == col.lower():
                                st.session_state.var_descriptions[col] = description
                                matched_count += 1
                                break
                            # Partial match (either direction)
                            elif (var_name.lower() in col.lower() or 
                                  col.lower() in var_name.lower()):
                                st.session_state.var_descriptions[col] = description
                                matched_count += 1
                                break
                            # Fuzzy matching for common variations
                            elif self._fuzzy_match_variables(var_name, col):
                                st.session_state.var_descriptions[col] = description
                                matched_count += 1
                                break
                    
                    st.success(f"✅ Matched {matched_count} variables!")
                    st.rerun()
            
            with col2:
                if st.button("🎯 Smart Match Experimental Variables"):
                    # Focus on experimental design variables
                    experimental_matches = 0
                    experimental_keywords = ['condition', 'treatment', 'group', 'randomiz', 'manipulation', 'assign']
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        # Check if column might be experimental
                        for keyword in experimental_keywords:
                            if keyword in col_lower:
                                # Find best matching PDF variable
                                best_match = self._find_best_experimental_match(col, pdf_suggestions, experimental_keywords)
                                if best_match:
                                    st.session_state.var_descriptions[col] = pdf_suggestions[best_match]
                                    experimental_matches += 1
                                    break
                    
                    if experimental_matches > 0:
                        st.success(f"🧪 Matched {experimental_matches} experimental variables!")
                        st.rerun()
                    else:
                        st.warning("No experimental variables detected in your data columns")
        
        variable_descriptions = {}
        
        for col in df.columns:
            # Enhanced variable description with PDF integration
            # Auto-detect variable characteristics
            var_info = detect_variable_type(df[col])
            
            # Get scale information from PDF if available
            pdf_scale_info = ""
            if col in scale_info or any(col.lower() in k.lower() for k in scale_info.keys()):
                for scale_var, scale_data in scale_info.items():
                    if col.lower() in scale_var.lower() or scale_var.lower() in col.lower():
                        if isinstance(scale_data, dict):
                            pdf_scale_info = f" (PDF detected: {scale_data.get('type', '')}"
                            if 'range' in scale_data:
                                pdf_scale_info += f", {scale_data['range']}"
                            pdf_scale_info += ")"
            
            # Create comprehensive placeholder
            if var_info['scale_type'] == 'binary':
                placeholder = f"Binary variable (0/1 or {var_info['categories']}){pdf_scale_info}"
            elif var_info['scale_type'].startswith('likert'):
                scale_range = var_info['scale_type'].split('_')[1:3]
                placeholder = f"{scale_range[0]}-{scale_range[1]} Likert scale{pdf_scale_info}"
            elif var_info['scale_type'] == 'categorical':
                placeholder = f"Categorical variable ({var_info['unique_count']} categories){pdf_scale_info}"
            else:
                placeholder = f"Numeric (range: {var_info['min_val']:.2f}-{var_info['max_val']:.2f}){pdf_scale_info}"
            
            # Enhanced PDF suggestion matching
            current_value = st.session_state.var_descriptions.get(col, "")
            
            if not current_value and pdf_suggestions:
                # Try multiple matching strategies
                matches = []
                
                # Exact match
                if col in pdf_suggestions:
                    matches.append((col, pdf_suggestions[col]))
                
                # Case-insensitive exact match
                for pdf_var, pdf_desc in pdf_suggestions.items():
                    if pdf_var.lower() == col.lower():
                        matches.append((pdf_var, pdf_desc))
                        break
                
                # Partial matching
                if not matches:
                    for pdf_var, pdf_desc in pdf_suggestions.items():
                        if (pdf_var.lower() in col.lower() or 
                            col.lower() in pdf_var.lower()):
                            matches.append((pdf_var, pdf_desc))
                
                # Experimental variable matching
                if not matches:
                    experimental_keywords = ['condition', 'treatment', 'group', 'randomiz', 'manipulation']
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in experimental_keywords):
                        for pdf_var, pdf_desc in pdf_suggestions.items():
                            if any(keyword in pdf_var.lower() for keyword in experimental_keywords):
                                matches.append((pdf_var, pdf_desc))
                                break
                
                # Use the best match
                if matches:
                    current_value = matches[0][1]
                    st.session_state.var_descriptions[col] = current_value
            
            description = st.text_input(
                f"Description for '{col}' ({var_info['scale_type']}):",
                value=current_value,
                key=f"desc_{col}",
                placeholder=placeholder,
                help=f"Auto-detected: {var_info['scale_type']} variable. {placeholder}"
            )
            
            if description:
                variable_descriptions[col] = description
                st.session_state.var_descriptions[col] = description

    def _fuzzy_match_variables(self, pdf_var: str, data_col: str) -> bool:
        """Enhanced fuzzy matching for variable names"""
        pdf_lower = pdf_var.lower()
        col_lower = data_col.lower()
        
        # Common experimental variable synonyms
        experimental_synonyms = {
            'condition': ['cond', 'treatment', 'group', 'condition_assignment'],
            'treatment': ['treat', 'condition', 'intervention', 'manipulation'],
            'group': ['grp', 'condition', 'assignment', 'cohort'],
            'randomization': ['random', 'rand', 'assignment', 'assign'],
            'manipulation': ['manip', 'treatment', 'condition']
        }
        
        # Check if either variable contains experimental synonyms
        for key, synonyms in experimental_synonyms.items():
            if key in pdf_lower:
                if any(syn in col_lower for syn in synonyms):
                    return True
            if key in col_lower:
                if any(syn in pdf_lower for syn in synonyms):
                    return True
        
        # Common abbreviations and variations
        common_variations = {
            'wtp': 'willingness_to_pay',
            'brandtrust': 'brand_trust',
            'cust': 'customer',
            'satisf': 'satisfaction',
            'demo': 'demographic'
        }
        
        for abbrev, full in common_variations.items():
            if (abbrev in pdf_lower and full in col_lower) or (abbrev in col_lower and full in pdf_lower):
                return True
        
        return False
    
    def _find_best_experimental_match(self, data_col: str, pdf_suggestions: Dict, experimental_keywords: List[str]) -> str:
        """Find the best matching experimental variable from PDF suggestions"""
        col_lower = data_col.lower()
        
        # Direct keyword matching
        for pdf_var in pdf_suggestions.keys():
            pdf_lower = pdf_var.lower()
            for keyword in experimental_keywords:
                if keyword in col_lower and keyword in pdf_lower:
                    return pdf_var
        
        # Fuzzy matching for experimental variables
        for pdf_var in pdf_suggestions.keys():
            if self._fuzzy_match_variables(pdf_var, data_col):
                return pdf_var
        
        return None
        
        # Variable selection
        st.subheader("🎯 Variable Selection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Dependent Variable (DV)**")
            dependent_var = st.selectbox(
                "Select the outcome variable:",
                options=df.columns,
                key="dependent_var_select"
            )
        
        with col2:
            st.markdown("**Independent Variables (IV)**")
            independent_vars = st.multiselect(
                "Select predictor variables:",
                options=[col for col in df.columns if col != dependent_var],
                key="independent_vars_select"
            )
        
        with col3:
            st.markdown("**Control Variables (CV)**")
            control_vars = st.multiselect(
                "Select control variables:",
                options=[col for col in df.columns if col != dependent_var and col not in independent_vars],
                key="control_vars_select"
            )
        
        # Store selections in session state with different keys
        st.session_state.selected_dependent_var = dependent_var
        st.session_state.selected_independent_vars = independent_vars
        st.session_state.selected_control_vars = control_vars
        
        # Show variable type analysis
        if dependent_var and independent_vars:
            st.subheader("🔍 Variable Type Analysis")
            
            analysis_cols = st.columns(3)
            
            with analysis_cols[0]:
                st.markdown("**Dependent Variable**")
                dv_info = detect_variable_type(df[dependent_var])
                if dv_info['scale_type'] == 'binary':
                    st.info(f"📊 Binary ({dv_info['categories']})")
                elif dv_info['scale_type'].startswith('likert'):
                    scale_parts = dv_info['scale_type'].split('_')
                    st.info(f"📊 Likert {scale_parts[1]}-{scale_parts[2]}")
                elif dv_info['scale_type'] == 'categorical':
                    st.info(f"📊 Categorical ({dv_info['unique_count']} categories)")
                else:
                    st.info(f"📊 Continuous ({dv_info['min_val']:.2f} to {dv_info['max_val']:.2f})")
            
            with analysis_cols[1]:
                st.markdown("**Independent Variables**")
                for iv in independent_vars:
                    iv_info = detect_variable_type(df[iv])
                    if iv_info['scale_type'] == 'binary':
                        st.success(f"{iv}: Binary")
                    elif iv_info['scale_type'].startswith('likert'):
                        scale_parts = iv_info['scale_type'].split('_')
                        st.success(f"{iv}: Likert {scale_parts[1]}-{scale_parts[2]}")
                    else:
                        st.success(f"{iv}: {iv_info['scale_type'].title()}")
            
            with analysis_cols[2]:
                st.markdown("**Control Variables**")
                for cv in control_vars:
                    cv_info = detect_variable_type(df[cv])
                    if cv_info['scale_type'] == 'binary':
                        st.warning(f"{cv}: Binary")
                    elif cv_info['scale_type'].startswith('likert'):
                        scale_parts = cv_info['scale_type'].split('_')
                        st.warning(f"{cv}: Likert {scale_parts[1]}-{scale_parts[2]}")
                    else:
                        st.warning(f"{cv}: {cv_info['scale_type'].title()}")
            
            st.subheader("📊 Preliminary Analysis")
            
            selected_vars = [dependent_var] + independent_vars + control_vars
            
            # Correlation heatmap
            fig_corr = create_correlation_heatmap(df, selected_vars)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Relationship plots
            fig_rel = create_relationship_plots(df, dependent_var, independent_vars)
            if fig_rel:
                st.plotly_chart(fig_rel, use_container_width=True)
    
    with tab4:
        st.header("🧠 AI-Powered Analysis")
        
        # Check if all required data is available
        if 'df' not in st.session_state:
            st.warning("Please upload data first")
            return
        
        if 'selected_dependent_var' not in st.session_state:
            st.warning("Please select variables first")
            return
        
        df = st.session_state.df
        dependent_var = st.session_state.selected_dependent_var
        independent_vars = st.session_state.get('selected_independent_vars', [])
        control_vars = st.session_state.get('selected_control_vars', [])
        variable_descriptions = st.session_state.get('var_descriptions', {})
        context = st.session_state.get('study_context', '')
        
        if st.button("🚀 Analyze Relationships", type="primary"):
            with st.spinner("Analyzing data with DeepSeek AI..."):
                analysis_result = deepseek_client.analyze_relationships(
                    df, dependent_var, independent_vars, control_vars, variable_descriptions, context
                )
                
                st.session_state.analysis_result = analysis_result
                
                # Extract suggested variables for later use
                suggested_vars = extract_suggested_variables(analysis_result)
                st.session_state.suggested_variables = suggested_vars
        
        # Display analysis results
        if 'analysis_result' in st.session_state:
            st.subheader("🔍 AI Analysis Results")
            
            # Create expandable sections for better readability
            with st.expander("📊 Full Analysis Report", expanded=True):
                st.markdown(st.session_state.analysis_result)
            
            # Extract and display key insights in a more digestible format
            analysis_text = st.session_state.analysis_result
            
            # Try to extract key findings
            if "KEY FINDINGS" in analysis_text or "ACTIONABLE RECOMMENDATIONS" in analysis_text:
                st.subheader("⭐ Key Takeaways")
                
                # Extract key sections
                lines = analysis_text.split('\n')
                key_section = False
                recommendations = []
                
                for line in lines:
                    if "ACTIONABLE RECOMMENDATIONS" in line.upper():
                        key_section = True
                        continue
                    elif line.startswith('#') and key_section:
                        break
                    elif key_section and line.strip():
                        recommendations.append(line.strip())
                
                if recommendations:
                    st.success("🎯 **What to do next:**")
                    for rec in recommendations[:3]:  # Show top 3 recommendations
                        if rec and not rec.startswith('#'):
                            st.write(f"• {rec}")
            
            # Display extracted suggestions as tags  
            if 'suggested_variables' in st.session_state and st.session_state.suggested_variables:
                st.subheader("💡 Suggested Variables for Future Research")
                cols = st.columns(min(4, len(st.session_state.suggested_variables)))
                for i, var in enumerate(st.session_state.suggested_variables):
                    with cols[i % len(cols)]:
                        st.info(f"🏷️ {var}")
    
    with tab5:
        st.header("🔮 Silicon Sampling & Simulation")
        
        if 'analysis_result' not in st.session_state:
            st.warning("Please complete the AI analysis first")
            return
        
        # Configuration section
        st.subheader("⚙️ Simulation Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_samples = st.number_input("Number of synthetic samples:", min_value=10, max_value=5000, value=200)
            
        with col2:
            file_format = st.selectbox("Export format:", ["csv", "json"], index=0)
        
        # Additional variables section
        st.subheader("➕ Include Additional Variables")
        
        additional_vars = []
        
        # Show AI suggested variables if available
        if 'suggested_variables' in st.session_state and st.session_state.suggested_variables:
            st.markdown("**🤖 AI-Suggested Variables (from your analysis):**")
            suggested = st.session_state.suggested_variables
            
            # Create checkboxes for suggested variables
            selected_suggestions = []
            if suggested:
                cols = st.columns(min(3, len(suggested)))
                for i, var in enumerate(suggested):
                    with cols[i % len(cols)]:
                        if st.checkbox(f"Include {var}", key=f"suggest_{i}"):
                            selected_suggestions.append(var)
            
            additional_vars = selected_suggestions
        else:
            st.info("💡 Complete the AI Analysis first to see variable suggestions here!")
        
        # Custom additional variables
        st.markdown("**Custom Additional Variables:**")
        custom_vars_input = st.text_input(
            "Enter custom variable names (comma-separated):",
            placeholder="e.g., customer_age, purchase_frequency, satisfaction_score"
        )
        
        if custom_vars_input:
            custom_vars = [var.strip() for var in custom_vars_input.split(',') if var.strip()]
            additional_vars.extend(custom_vars)
        
        # Display selected additional variables
        if additional_vars:
            st.success(f"✅ Additional variables to include: {', '.join(additional_vars)}")
        
        # Generate synthetic data
        st.subheader("🎲 Generate Synthetic Data")
        
        if st.button("🚀 Generate Silicon Sample", type="primary"):
            with st.spinner("Generating synthetic data based on your experiment..."):
                df = st.session_state.df
                dependent_var = st.session_state.selected_dependent_var
                independent_vars = st.session_state.get('selected_independent_vars', [])
                control_vars = st.session_state.get('selected_control_vars', [])
                
                # Generate synthetic data
                synthetic_df = deepseek_client.generate_synthetic_data(
                    df, dependent_var, independent_vars, control_vars, additional_vars, n_samples
                )
                
                st.session_state.synthetic_data = synthetic_df
                st.session_state.synthetic_config = {
                    'n_samples': n_samples,
                    'additional_vars': additional_vars,
                    'file_format': file_format
                }
        
        # Display and download synthetic data
        if 'synthetic_data' in st.session_state:
            synthetic_df = st.session_state.synthetic_data
            config = st.session_state.synthetic_config
            
            st.subheader("📊 Generated Synthetic Data")
            st.success(f"✅ Successfully generated {len(synthetic_df)} synthetic samples!")
            
            # Show preview
            st.markdown("**Data Preview:**")
            st.dataframe(synthetic_df.head(10))
            
            # Show statistics comparison
            st.subheader("📈 Original vs Synthetic Comparison")
            
            original_df = st.session_state.df
            comparison_vars = [col for col in synthetic_df.columns if col in original_df.columns]
            
            if comparison_vars:
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.markdown("**Original Data Statistics:**")
                    st.dataframe(original_df[comparison_vars].describe())
                
                with comp_col2:
                    st.markdown("**Synthetic Data Statistics:**")
                    st.dataframe(synthetic_df[comparison_vars].describe())
            
            # Visualization comparison
            if len(comparison_vars) >= 2:
                st.subheader("📊 Distribution Comparison")
                
                # Select variables for comparison
                comp_var1 = st.selectbox("Select first variable for comparison:", comparison_vars)
                comp_var2 = st.selectbox("Select second variable for comparison:", 
                                       [v for v in comparison_vars if v != comp_var1])
                
                if comp_var1 and comp_var2:
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=["Original Data", "Synthetic Data"],
                        horizontal_spacing=0.1
                    )
                    
                    # Original data scatter
                    fig.add_trace(
                        go.Scatter(
                            x=original_df[comp_var1],
                            y=original_df[comp_var2],
                            mode='markers',
                            name='Original',
                            marker=dict(color='blue', opacity=0.6)
                        ),
                        row=1, col=1
                    )
                    
                    # Synthetic data scatter
                    fig.add_trace(
                        go.Scatter(
                            x=synthetic_df[comp_var1],
                            y=synthetic_df[comp_var2],
                            mode='markers',
                            name='Synthetic',
                            marker=dict(color='red', opacity=0.6)
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        title=f"Distribution Comparison: {comp_var1} vs {comp_var2}",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Download section
            st.subheader("📥 Download Synthetic Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_download_button(synthetic_df, "synthetic_data", "csv")
            
            with col2:
                create_download_button(synthetic_df, "synthetic_data", "json")
            
            with col3:
                # Create analysis report
                report_data = {
                    "experiment_info": {
                        "dependent_variable": st.session_state.selected_dependent_var,
                        "independent_variables": st.session_state.get('selected_independent_vars', []),
                        "control_variables": st.session_state.get('selected_control_vars', []),
                        "additional_variables": config['additional_vars'],
                        "original_sample_size": len(st.session_state.df),
                        "synthetic_sample_size": config['n_samples']
                    },
                    "analysis_summary": st.session_state.get('analysis_result', ''),
                    "generation_timestamp": pd.Timestamp.now().isoformat()
                }
                
                report_json = json.dumps(report_data, indent=2)
                st.download_button(
                    label="📋 Download Analysis Report",
                    data=report_json,
                    file_name="silicon_sampling_report.json",
                    mime="application/json"
                )
        
        # Future features and methodology
        st.subheader("🔬 Silicon Sampling Methodology")
        with st.expander("Learn more about Silicon Sampling"):
            st.markdown("""
            **Silicon Sampling** is a novel approach to experimental replication and extension that leverages AI to:
            
            1. **Analyze Relationships**: Deep understanding of variable interactions in your original data
            2. **Model Replication**: Generate synthetic data that preserves the statistical properties of your experiment
            3. **Hypothesis Extension**: Explore additional variables and scenarios without collecting new data
            4. **Risk Mitigation**: Test experimental variations before conducting expensive real-world studies
            
            **Key Benefits:**
            - 🚀 **Speed**: Generate new experimental scenarios in minutes
            - 💰 **Cost-Effective**: Reduce need for expensive data collection
            - 🔍 **Exploration**: Test hypotheses with additional variables
            - 📊 **Validation**: Compare synthetic vs. real data distributions
            
            **Use Cases:**
            - Pilot study expansion
            - Hypothesis testing before full experiments
            - Training data augmentation
            - Scenario modeling and what-if analysis
            """)
        
        # Advanced features placeholder
        st.subheader("🚧 Advanced Features (Coming Soon)")
        st.markdown("""
        - **🎯 Targeted Sampling**: Generate data for specific population segments
        - **⚖️ Bias Detection**: Identify and correct sampling biases
        - **🔄 Iterative Refinement**: Improve synthetic data quality through feedback loops
        - **🧪 A/B Test Simulation**: Simulate experimental conditions and power analysis
        - **📈 Longitudinal Modeling**: Generate time-series synthetic data
        - **🌐 Multi-modal Integration**: Combine different data types (text, images, numerical)
        """)

if __name__ == "__main__":
    main()
