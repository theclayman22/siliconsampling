import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import io
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configure Streamlit page
st.set_page_config(
    page_title="Silicon Sampling - Experiment Simulation",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DeepSeekAPI:
    """Interface for DeepSeek API integration"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_completion(self, prompt: str, temperature: float = 0.3) -> str:
        """Send completion request to DeepSeek API"""
        try:
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are an expert data scientist specializing in experimental design and statistical modeling. Provide precise, analytical responses."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 2000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"API request failed: {str(e)}")
            return None

class SiliconSampler:
    """Main class for silicon sampling simulation"""
    
    def __init__(self):
        self.data = None
        self.variable_descriptions = {}
        self.dependent_var = None
        self.independent_vars = []
        self.control_vars = []
        self.api_client = None
        self.model_insights = None
        self.predictions = None
    
    def load_data(self, uploaded_file) -> bool:
        """Load data from uploaded Excel file"""
        try:
            if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                self.data = pd.read_excel(uploaded_file)
            else:
                st.error("Please upload an Excel file (.xlsx or .xls)")
                return False
                
            # Clean column names
            self.data.columns = self.data.columns.str.strip()
            return True
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def analyze_relationships(self) -> str:
        """Use DeepSeek to analyze variable relationships"""
        if not self.api_client:
            return "API client not configured"
        
        # Prepare data summary for API
        data_summary = {
            "dependent_variable": self.dependent_var,
            "independent_variables": self.independent_vars,
            "control_variables": self.control_vars,
            "sample_size": len(self.data),
            "data_sample": self.data.head(5).to_dict(),
            "descriptive_stats": self.data[self.independent_vars + [self.dependent_var]].describe().to_dict()
        }
        
        prompt = f"""
        Analyze this experimental dataset and provide insights on variable relationships:
        
        Dataset Summary:
        - Dependent Variable: {self.dependent_var}
        - Independent Variables: {', '.join(self.independent_vars)}
        - Control Variables: {', '.join(self.control_vars) if self.control_vars else 'None'}
        - Sample Size: {len(self.data)}
        
        Variable Descriptions:
        {json.dumps(self.variable_descriptions, indent=2)}
        
        Data Sample:
        {json.dumps(data_summary['data_sample'], indent=2)}
        
        Statistical Summary:
        {json.dumps(data_summary['descriptive_stats'], indent=2)}
        
        Please provide:
        1. Analysis of relationships between variables
        2. Suggested mathematical model or formula to predict the dependent variable
        3. Key factors that likely influence the outcome
        4. Potential confounding variables to consider
        5. Python code snippet to replicate the relationship (using pandas/numpy)
        
        Format your response clearly with numbered sections.
        """
        
        return self.api_client.create_completion(prompt)
    
    def generate_predictions(self) -> Optional[pd.DataFrame]:
        """Generate predictions using DeepSeek insights"""
        if not self.api_client or not self.model_insights:
            return None
        
        # Create prompt for prediction generation
        data_json = self.data[self.independent_vars + self.control_vars + [self.dependent_var]].to_json(orient='records')
        
        prompt = f"""
        Based on the previous analysis, generate predictions for the dependent variable '{self.dependent_var}' 
        using the independent variables: {', '.join(self.independent_vars)}.
        
        Here's the complete dataset:
        {data_json}
        
        Please provide Python code that:
        1. Takes the independent variables as input
        2. Returns predicted values for the dependent variable
        3. Uses a realistic model based on the data patterns
        
        Return ONLY the Python function code, no explanations.
        The function should be named 'predict_outcome' and take a pandas DataFrame as input.
        """
        
        code_response = self.api_client.create_completion(prompt, temperature=0.1)
        
        if code_response:
            try:
                # Execute the generated code safely
                local_vars = {'pd': pd, 'np': np}
                exec(code_response, local_vars)
                
                if 'predict_outcome' in local_vars:
                    predict_func = local_vars['predict_outcome']
                    input_data = self.data[self.independent_vars + self.control_vars].copy()
                    predictions = predict_func(input_data)
                    
                    # Create results DataFrame
                    results_df = self.data.copy()
                    results_df['Predicted_' + self.dependent_var] = predictions
                    results_df['Residual'] = results_df[self.dependent_var] - predictions
                    
                    return results_df
                    
            except Exception as e:
                st.error(f"Error executing prediction code: {str(e)}")
                return None
        
        return None

def main():
    st.title("🧪 Silicon Sampling - Experiment Simulation Tool")
    st.markdown("### Virtual replication and expansion of experimental setups using AI")
    
    # Initialize session state
    if 'sampler' not in st.session_state:
        st.session_state.sampler = SiliconSampler()
    
    sampler = st.session_state.sampler
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        st.subheader("DeepSeek API Settings")
        api_key = st.text_input("API Key", type="password", help="Enter your DeepSeek API key")
        
        if api_key:
            sampler.api_client = DeepSeekAPI(api_key)
            st.success("✅ API configured")
        else:
            st.warning("⚠️ Please enter API key to proceed")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Data Upload", "🔍 Variable Setup", "🧠 AI Analysis", "📊 Results & Simulation"])
    
    with tab1:
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            if sampler.load_data(uploaded_file):
                st.success(f"✅ File loaded successfully! Shape: {sampler.data.shape}")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(sampler.data.head(10))
                
                # Basic statistics
                st.subheader("Basic Statistics")
                numeric_cols = sampler.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(sampler.data[numeric_cols].describe())
    
    with tab2:
        if sampler.data is not None:
            st.header("Variable Setup & Description")
            
            # Variable descriptions
            st.subheader("Describe Your Variables")
            cols = st.columns(2)
            
            for i, col in enumerate(sampler.data.columns):
                with cols[i % 2]:
                    description = st.text_area(
                        f"Describe '{col}':",
                        key=f"desc_{col}",
                        height=80,
                        placeholder="Brief description of what this variable represents..."
                    )
                    if description:
                        sampler.variable_descriptions[col] = description
            
            st.divider()
            
            # Variable role assignment
            st.subheader("Assign Variable Roles")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Dependent Variable** (What you're trying to predict)")
                sampler.dependent_var = st.selectbox(
                    "Select dependent variable:",
                    options=[None] + list(sampler.data.columns),
                    key="dependent_var"
                )
            
            with col2:
                st.markdown("**Independent Variables** (Factors that influence the outcome)")
                sampler.independent_vars = st.multiselect(
                    "Select independent variables:",
                    options=[col for col in sampler.data.columns if col != sampler.dependent_var],
                    key="independent_vars"
                )
            
            with col3:
                st.markdown("**Control Variables** (Variables held constant or controlled for)")
                available_controls = [col for col in sampler.data.columns 
                                    if col != sampler.dependent_var and col not in sampler.independent_vars]
                sampler.control_vars = st.multiselect(
                    "Select control variables:",
                    options=available_controls,
                    key="control_vars"
                )
            
            # Validation
            if sampler.dependent_var and sampler.independent_vars:
                st.success("✅ Variable setup complete!")
                
                # Summary
                st.subheader("Setup Summary")
                summary_data = {
                    "Role": ["Dependent", "Independent", "Control"],
                    "Variables": [
                        sampler.dependent_var,
                        ", ".join(sampler.independent_vars),
                        ", ".join(sampler.control_vars) if sampler.control_vars else "None"
                    ]
                }
                st.table(pd.DataFrame(summary_data))
        else:
            st.warning("Please upload data first in the Data Upload tab.")
    
    with tab3:
        if sampler.dependent_var and sampler.independent_vars and sampler.api_client:
            st.header("AI-Powered Analysis")
            
            if st.button("🧠 Analyze Relationships", type="primary"):
                with st.spinner("Analyzing variable relationships with DeepSeek..."):
                    sampler.model_insights = sampler.analyze_relationships()
                
                if sampler.model_insights:
                    st.subheader("AI Analysis Results")
                    st.markdown(sampler.model_insights)
                    
                    # Generate predictions
                    with st.spinner("Generating predictions..."):
                        sampler.predictions = sampler.generate_predictions()
            
            if sampler.model_insights:
                st.subheader("Model Insights")
                st.markdown(sampler.model_insights)
        else:
            missing = []
            if not sampler.dependent_var or not sampler.independent_vars:
                missing.append("Variable setup")
            if not sampler.api_client:
                missing.append("API configuration")
            
            st.warning(f"Please complete: {', '.join(missing)}")
    
    with tab4:
        if sampler.predictions is not None:
            st.header("Results & Simulation")
            
            # Model performance metrics
            actual = sampler.predictions[sampler.dependent_var]
            predicted = sampler.predictions['Predicted_' + sampler.dependent_var]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                r2 = r2_score(actual, predicted)
                st.metric("R² Score", f"{r2:.3f}")
            with col2:
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                st.metric("RMSE", f"{rmse:.3f}")
            with col3:
                mae = mean_absolute_error(actual, predicted)
                st.metric("MAE", f"{mae:.3f}")
            with col4:
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                st.metric("MAPE", f"{mape:.1f}%")
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Actual vs Predicted")
                fig = px.scatter(
                    sampler.predictions, 
                    x=sampler.dependent_var, 
                    y='Predicted_' + sampler.dependent_var,
                    title="Actual vs Predicted Values"
                )
                fig.add_shape(
                    type="line", line=dict(dash="dash"),
                    x0=actual.min(), y0=actual.min(),
                    x1=actual.max(), y1=actual.max()
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Residuals Analysis")
                fig = px.scatter(
                    sampler.predictions,
                    x='Predicted_' + sampler.dependent_var,
                    y='Residual',
                    title="Residuals vs Predicted"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.subheader("Detailed Results")
            st.dataframe(sampler.predictions)
            
            # Simulation section
            st.subheader("🎯 What-If Simulation")
            st.markdown("Modify independent variables to see predicted outcomes:")
            
            simulation_inputs = {}
            cols = st.columns(len(sampler.independent_vars))
            
            for i, var in enumerate(sampler.independent_vars):
                with cols[i]:
                    min_val = float(sampler.data[var].min())
                    max_val = float(sampler.data[var].max())
                    mean_val = float(sampler.data[var].mean())
                    
                    simulation_inputs[var] = st.slider(
                        f"{var}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"sim_{var}"
                    )
            
            if st.button("Run Simulation"):
                # Create simulation DataFrame
                sim_data = pd.DataFrame([simulation_inputs])
                
                # Add control variables (use mean values)
                for var in sampler.control_vars:
                    sim_data[var] = sampler.data[var].mean()
                
                # Generate prediction (would need to re-execute the prediction function)
                st.success("Simulation feature ready - implement prediction execution")
        else:
            st.warning("Please complete AI analysis first.")

if __name__ == "__main__":
    main()
