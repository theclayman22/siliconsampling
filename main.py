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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

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
                            variable_descriptions: Dict[str, str]) -> Dict[str, Any]:
        """Analyze relationships between variables using DeepSeek API"""
        
        # Prepare data summary for API
        data_summary = {
            "dependent_variable": dependent_var,
            "independent_variables": independent_vars,
            "control_variables": control_vars,
            "data_shape": data.shape,
            "variable_descriptions": variable_descriptions,
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

        Please provide:
        1. Analysis of the relationship between the dependent variable ({dependent_var}) and independent variables ({independent_vars})
        2. How control variables ({control_vars}) might influence these relationships
        3. A mathematical model or equation that could describe these relationships
        4. Suggestions for additional variables that might be relevant to include in future experiments
        5. Potential confounding factors or limitations in the current data

        Format your response as a structured analysis with clear sections.
        """
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 2000
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
                              control_vars: List[str], n_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic data based on learned relationships"""
        
        prompt = f"""
        Based on the data analysis, generate a Python function that can create synthetic data 
        replicating the relationships between:
        - Dependent variable: {dependent_var}
        - Independent variables: {independent_vars}
        - Control variables: {control_vars}

        The function should:
        1. Generate realistic values for independent and control variables
        2. Calculate the dependent variable based on the relationships you identified
        3. Add appropriate noise/uncertainty
        4. Return a pandas DataFrame with {n_samples} rows

        Provide only the Python function code, no explanation.
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
                return response.json()['choices'][0]['message']['content']
            else:
                return f"API Error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"

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
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Data Upload", "🔍 Variable Selection", "🧠 AI Analysis", "🔮 Simulation"])
    
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
    
    with tab3:
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
        
        if st.button("🚀 Analyze Relationships", type="primary"):
            with st.spinner("Analyzing data with DeepSeek AI..."):
                analysis_result = deepseek_client.analyze_relationships(
                    df, dependent_var, independent_vars, control_vars, variable_descriptions
                )
                
                st.session_state.analysis_result = analysis_result
        
        # Display analysis results
        if 'analysis_result' in st.session_state:
            st.subheader("🔍 AI Analysis Results")
            st.markdown(st.session_state.analysis_result)
    
    with tab4:
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
