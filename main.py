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
        if 'suggested_variables' in st.session_state:
            st.markdown("**AI-Suggested Variables:**")
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
        
        # Custom additional variables
        st.markdown("**Custom Additional Variables:**")
        custom_vars_input = st.text_input(
            "Enter custom variable names (comma-separated):",
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
        As an expert data scientist, analyze the following experimental data and provide insights:

        Data Summary:
        {json.dumps(data_summary, indent=2)}

        Variable Descriptions:
        {json.dumps(variable_descriptions, indent=2)}

        Additional Context:
        {context}

        Please provide a structured analysis with the following sections:

        ## 1. RELATIONSHIP ANALYSIS
        Analyze the relationship between the dependent variable ({dependent_var}) and independent variables ({independent_vars}).

        ## 2. REGRESSION MODEL
        Provide a clear mathematical formulation:
        - Linear regression equation in the form: Y = β₀ + β₁X₁ + β₂X₂ + ... + ε
        - Expected coefficient signs and magnitudes
        - R² estimation

        ## 3. CONTROL VARIABLES IMPACT
        How control variables ({control_vars}) might influence these relationships.

        ## 4. SUGGESTED ADDITIONAL VARIABLES
        List 5-8 specific variables that could enhance this study, with brief justifications:
        - Variable Name: Brief Description (Rationale)

        ## 5. POTENTIAL LIMITATIONS
        Identify potential confounding factors or data limitations.

        ## 6. SYNTHETIC DATA PARAMETERS
        Suggest realistic parameter ranges for generating synthetic data:
        - Coefficient estimates
        - Error term variance
        - Variable distributions

        Format your response with clear markdown headers and bullet points.
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
        """Create synthetic dataset using AI-generated parameters"""
        
        synthetic_data = {}
        
        # Generate independent and control variables first
        for var in variables:
            if var in original_data.columns:
                if original_data[var].dtype in ['int64', 'float64']:
                    mean = original_data[var].mean()
                    std = original_data[var].std()
                    synthetic_data[var] = np.random.normal(mean, std, n_samples)
                else:
                    # For categorical variables, sample from original distribution
                    synthetic_data[var] = np.random.choice(original_data[var].dropna(), n_samples)
        
        # Generate dependent variable using coefficients if available
        if 'coefficients' in params and variables:
            dependent_var = None
            for var in variables:
                if var in original_data.columns and original_data[var].dtype in ['int64', 'float64']:
                    if not dependent_var:  # Assume first numeric variable is dependent
                        dependent_var = var
                        break
            
            if dependent_var and dependent_var in synthetic_data:
                # Simple linear combination
                y = params['coefficients'].get('intercept', 0)
                for var, coef in params['coefficients'].items():
                    if var != 'intercept' and var in synthetic_data:
                        y += coef * synthetic_data[var]
                
                # Add noise
                noise = np.random.normal(0, params.get('error_variance', 1), n_samples)
                synthetic_data[dependent_var] = y + noise
        
        return pd.DataFrame(synthetic_data)
    
    def _create_fallback_synthetic_data(self, original_data: pd.DataFrame, 
                                      variables: List[str], n_samples: int) -> pd.DataFrame:
        """Fallback method to create synthetic data when AI fails"""
        
        synthetic_data = {}
        
        for var in variables:
            if var in original_data.columns:
                if original_data[var].dtype in ['int64', 'float64']:
                    mean = original_data[var].mean()
                    std = original_data[var].std()
                    synthetic_data[var] = np.random.normal(mean, std, n_samples)
                else:
                    synthetic_data[var] = np.random.choice(original_data[var].dropna(), n_samples)
        
        return pd.DataFrame(synthetic_data)

    def analyze_pdf_document(self, pdf_content: bytes) -> str:
        """Analyze PDF document for variable context and suggestions"""
        
        if not PDF_AVAILABLE:
            return "PDF analysis not available. Please install PyPDF2."
        
        try:
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Truncate text if too long (API limits)
            if len(text) > 8000:
                text = text[:8000] + "..."
            
            prompt = f"""
            Analyze this research document/codebook/survey and provide:
            
            ## DOCUMENT ANALYSIS
            
            Document text:
            {text}
            
            Please provide:
            
            ## 1. VARIABLE CONTEXT SUGGESTIONS
            Based on the document, suggest what each variable mentioned might represent and its role in the study.
            
            ## 2. STUDY CONTEXT
            Provide a brief summary of the research context, methodology, and objectives.
            
            ## 3. VARIABLE RELATIONSHIPS
            Suggest likely relationships between variables based on the research design.
            
            ## 4. ADDITIONAL VARIABLES TO CONSIDER
            Based on the research domain, suggest additional variables that might be relevant.
            
            ## 5. METHODOLOGICAL CONSIDERATIONS
            Any special considerations for analysis based on the study design.
            
            Format with clear markdown headers.
            """
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 2500
                }
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"Error analyzing PDF: {response.status_code}"
                
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

def extract_suggested_variables(analysis_text: str) -> List[str]:
    """Extract suggested variables from AI analysis"""
    variables = []
    
    # Look for suggested variables section
    lines = analysis_text.split('\n')
    in_variables_section = False
    
    for line in lines:
        if 'SUGGESTED' in line.upper() and 'VARIABLE' in line.upper():
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
                    if len(var_name) < 50:  # Reasonable variable name length
                        variables.append(var_name)
    
    return variables[:8]  # Limit to 8 suggestions

def create_download_button(data: pd.DataFrame, filename: str, file_format: str = 'csv'):
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
    """Load and preview uploaded Excel file"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def load_and_preview_data(uploaded_file) -> pd.DataFrame:
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
                        pdf_analysis = deepseek_client.analyze_pdf_document(pdf_content)
                        st.session_state.pdf_analysis = pdf_analysis
                
                # Display PDF analysis results
                if 'pdf_analysis' in st.session_state:
                    st.subheader("📄 PDF Analysis Results")
                    st.markdown(st.session_state.pdf_analysis)
        else:
            st.warning("PDF analysis not available. Install PyPDF2 to enable this feature.")
            st.code("pip install PyPDF2", language="bash")
    
    with tab3:
        st.header("Variable Selection and Description")
        
        if 'df' not in st.session_state:
            st.warning("Please upload data first in the Data Upload tab")
            return
        
        df = st.session_state.df
        
        # Variable descriptions
        st.subheader("📝 Variable Descriptions")
        st.markdown("Provide brief descriptions for your variables to help the AI understand the context:")
        
        # Initialize variable descriptions in session state if not present
        if 'var_descriptions' not in st.session_state:
            st.session_state.var_descriptions = {}
        
        variable_descriptions = {}
        
        for col in df.columns:
            description = st.text_input(
                f"Description for '{col}':",
                value=st.session_state.var_descriptions.get(col, ""),
                key=f"desc_{col}",
                placeholder=f"Brief description of {col}..."
            )
            if description:
                variable_descriptions[col] = description
                st.session_state.var_descriptions[col] = description
        
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
        
        # Show correlation analysis
        if dependent_var and independent_vars:
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
            st.markdown(st.session_state.analysis_result)
            
            # Display extracted suggestions as tags
            if 'suggested_variables' in st.session_state and st.session_state.suggested_variables:
                st.subheader("💡 Extracted Variable Suggestions")
                cols = st.columns(4)
                for i, var in enumerate(st.session_state.suggested_variables):
                    with cols[i % 4]:
                        st.info(f"🏷️ {var}")
    
    with tab5:
        st.header("🔮 Experiment Simulation")
        
        if 'analysis_result' not in st.session_state:
            st.warning("Please complete the AI analysis first")
            return
        
        st.subheader("Generate Synthetic Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_samples = st.number_input("Number of synthetic samples:", min_value=10, max_value=1000, value=100)
        
        with col2:
            if st.button("🎲 Generate Synthetic Data"):
                with st.spinner("Generating synthetic data..."):
                    df = st.session_state.df
                    dependent_var = st.session_state.selected_dependent_var
                    independent_vars = st.session_state.get('selected_independent_vars', [])
                    control_vars = st.session_state.get('selected_control_vars', [])
                    
                    synthetic_code = deepseek_client.generate_synthetic_data(
                        df, dependent_var, independent_vars, control_vars, n_samples
                    )
                    
                    st.session_state.synthetic_code = synthetic_code
        
        # Display synthetic data generation code
        if 'synthetic_code' in st.session_state:
            st.subheader("📋 Generated Code")
            st.code(st.session_state.synthetic_code, language="python")
            
            st.info("⚠️ Note: The generated code should be reviewed and potentially modified before execution. This is a proof-of-concept for the cognitive layer approach.")
        
        # Future features placeholder
        st.subheader("🚧 Coming Soon")
        st.markdown("""
        - **Automated model validation**: Compare synthetic vs. original data
        - **Interactive parameter tuning**: Adjust model parameters in real-time
        - **Experiment extension**: Add new variables and test hypotheses
        - **Export capabilities**: Download results and generated code
        - **Advanced visualizations**: 3D plots and interactive dashboards
        """)

if __name__ == "__main__":
    main()
