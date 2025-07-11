import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import toml
from datetime import datetime
import requests
from typing import Dict, List, Any
import PyPDF2
import io
import time
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go

# Configuration
st.set_page_config(
    page_title="Social Science Experiment Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class SampleCharacteristics:
    """Data class to store sample characteristics"""
    age_range: str
    gender_distribution: str
    education_level: str
    socioeconomic_status: str
    cultural_background: str
    additional_traits: str

@dataclass
class Question:
    """Data class to store questionnaire questions"""
    id: str
    text: str
    type: str  # likert, multiple_choice, open_ended, etc.
    options: List[str]
    scale_range: tuple = None

class DeepSeekAPI:
    """DeepSeek API integration class"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using DeepSeek API"""
        try:
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"

class PDFProcessor:
    """PDF processing and questionnaire extraction"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    @staticmethod
    def parse_questionnaire(text: str) -> List[Question]:
        """Parse questionnaire text and extract questions"""
        questions = []
        
        # Simple regex patterns to identify questions
        question_patterns = [
            r'^\d+\.\s*(.+?)(?=\n\d+\.|$)',  # Numbered questions
            r'Q\d+[:\.]?\s*(.+?)(?=\nQ\d+|$)',  # Q1: format
            r'Question\s+\d+[:\.]?\s*(.+?)(?=\nQuestion\s+\d+|$)',  # Question 1: format
        ]
        
        # Look for Likert scale indicators
        likert_indicators = [
            r'strongly disagree.*strongly agree',
            r'never.*always',
            r'1.*5',
            r'1.*7',
            r'scale of \d+ to \d+'
        ]
        
        # Split text into potential questions
        lines = text.split('\n')
        current_question = ""
        question_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new question
            if re.match(r'^\d+\.|^Q\d+|^Question\s+\d+', line):
                if current_question:
                    # Process previous question
                    question_type = "open_ended"
                    options = []
                    
                    # Check for Likert scale
                    for pattern in likert_indicators:
                        if re.search(pattern, current_question.lower()):
                            question_type = "likert"
                            options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
                            break
                    
                    # Check for multiple choice (a), b), c) format)
                    if re.search(r'[a-z]\)', current_question.lower()):
                        question_type = "multiple_choice"
                        options = re.findall(r'[a-z]\)\s*([^a-z\)]+)', current_question.lower())
                    
                    questions.append(Question(
                        id=f"Q{question_counter}",
                        text=current_question.strip(),
                        type=question_type,
                        options=options
                    ))
                    question_counter += 1
                
                current_question = line
            else:
                current_question += " " + line
        
        # Process last question
        if current_question:
            question_type = "open_ended"
            options = []
            
            for pattern in likert_indicators:
                if re.search(pattern, current_question.lower()):
                    question_type = "likert"
                    options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
                    break
            
            if re.search(r'[a-z]\)', current_question.lower()):
                question_type = "multiple_choice"
                options = re.findall(r'[a-z]\)\s*([^a-z\)]+)', current_question.lower())
            
            questions.append(Question(
                id=f"Q{question_counter}",
                text=current_question.strip(),
                type=question_type,
                options=options
            ))
        
        return questions

class ExperimentSimulator:
    """Main experiment simulation class"""
    
    def __init__(self, api_key: str):
        self.deepseek_api = DeepSeekAPI(api_key)
    
    def generate_participant_prompt(self, sample_chars: SampleCharacteristics, participant_id: int) -> str:
        """Generate a prompt for a specific participant"""
        prompt = f"""
        You are simulating a research participant in a social science study. Here are your characteristics:
        
        Participant ID: {participant_id}
        Age Range: {sample_chars.age_range}
        Gender: {sample_chars.gender_distribution}
        Education Level: {sample_chars.education_level}
        Socioeconomic Status: {sample_chars.socioeconomic_status}
        Cultural Background: {sample_chars.cultural_background}
        Additional Traits: {sample_chars.additional_traits}
        
        Please embody these characteristics consistently throughout your responses. Answer questions as this person would, considering their background, values, and experiences. Be realistic and authentic in your responses.
        
        When answering:
        1. Stay in character based on the demographics provided
        2. Show some individual variation while maintaining demographic consistency
        3. For Likert scale questions, respond with the exact scale option (e.g., "Strongly Agree")
        4. For multiple choice questions, select the most appropriate option
        5. For open-ended questions, provide responses that reflect your demographic profile
        
        Remember: You are not Claude, you are a research participant with the specified characteristics.
        """
        return prompt
    
    def simulate_responses(self, questions: List[Question], sample_chars: SampleCharacteristics, 
                          num_participants: int) -> pd.DataFrame:
        """Simulate responses from multiple participants"""
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(num_participants):
            participant_responses = {"participant_id": i + 1}
            
            # Generate participant prompt
            participant_prompt = self.generate_participant_prompt(sample_chars, i + 1)
            
            status_text.text(f"Simulating participant {i + 1}/{num_participants}...")
            
            for question in questions:
                # Create question-specific prompt
                question_prompt = f"""
                {participant_prompt}
                
                Now please answer this question:
                {question.text}
                
                Question Type: {question.type}
                {f"Options: {', '.join(question.options)}" if question.options else ""}
                
                Please provide only your answer without explanation or meta-commentary.
                """
                
                # Get response from DeepSeek
                response = self.deepseek_api.generate_response(question_prompt, max_tokens=200)
                participant_responses[question.id] = response.strip()
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            results.append(participant_responses)
            progress_bar.progress((i + 1) / num_participants)
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(results)

def load_api_key():
    """Load API key from Streamlit secrets"""
    try:
        return st.secrets["DeepSeek_API_KEY01"]
    except:
        st.error("DeepSeek API key not found in secrets. Please configure DeepSeek_API_KEY01 in your secrets.toml file.")
        return None

def main():
    """Main Streamlit application"""
    
    st.title("🧠 Social Science Experiment Simulator")
    st.markdown("*Simulate social science experiments using DeepSeek as a cognitive layer*")
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        st.stop()
    
    # Initialize session state
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Sample Characteristics Section
    st.sidebar.subheader("Sample Characteristics")
    
    age_range = st.sidebar.text_input("Age Range", "18-65 years")
    gender_dist = st.sidebar.text_input("Gender Distribution", "50% Male, 50% Female")
    education = st.sidebar.text_input("Education Level", "Bachelor's degree or higher")
    ses = st.sidebar.text_input("Socioeconomic Status", "Middle class")
    cultural_bg = st.sidebar.text_input("Cultural Background", "Western, urban")
    additional_traits = st.sidebar.text_area("Additional Traits", "Tech-savvy, environmentally conscious")
    
    sample_chars = SampleCharacteristics(
        age_range=age_range,
        gender_distribution=gender_dist,
        education_level=education,
        socioeconomic_status=ses,
        cultural_background=cultural_bg,
        additional_traits=additional_traits
    )
    
    # Number of participants
    num_participants = st.sidebar.number_input("Number of Participants", min_value=1, max_value=100, value=10)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Questionnaire Upload")
        
        uploaded_file = st.file_uploader("Upload PDF Questionnaire", type=['pdf'])
        
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                # Extract text from PDF
                pdf_text = PDFProcessor.extract_text_from_pdf(uploaded_file)
                
                if pdf_text:
                    st.success("PDF text extracted successfully!")
                    
                    # Show extracted text (first 500 chars)
                    with st.expander("View Extracted Text (Preview)"):
                        st.text(pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text)
                    
                    # Parse questionnaire
                    questions = PDFProcessor.parse_questionnaire(pdf_text)
                    st.session_state.questions = questions
                    
                    st.success(f"Found {len(questions)} questions in the questionnaire")
                    
                    # Display questions
                    st.subheader("Extracted Questions")
                    for i, question in enumerate(questions):
                        with st.expander(f"Question {i+1}: {question.id}"):
                            st.write(f"**Text:** {question.text}")
                            st.write(f"**Type:** {question.type}")
                            if question.options:
                                st.write(f"**Options:** {', '.join(question.options)}")
    
    with col2:
        st.header("🎯 Simulation Results")
        
        if st.session_state.questions:
            if st.button("🚀 Run Simulation", type="primary"):
                simulator = ExperimentSimulator(api_key)
                
                with st.spinner("Running simulation..."):
                    results = simulator.simulate_responses(
                        st.session_state.questions,
                        sample_chars,
                        num_participants
                    )
                    st.session_state.results = results
                
                st.success("Simulation completed!")
        
        # Display results
        if st.session_state.results is not None:
            st.subheader("📊 Results")
            
            # Show raw data
            with st.expander("Raw Data"):
                st.dataframe(st.session_state.results)
            
            # Analysis section
            st.subheader("📈 Analysis")
            
            # For Likert scale questions, show distribution
            likert_questions = [q for q in st.session_state.questions if q.type == "likert"]
            
            if likert_questions:
                selected_question = st.selectbox(
                    "Select Likert Scale Question for Analysis",
                    [q.id for q in likert_questions]
                )
                
                if selected_question:
                    question_data = st.session_state.results[selected_question]
                    
                    # Create distribution chart
                    value_counts = question_data.value_counts()
                    
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Response Distribution for {selected_question}",
                        labels={"x": "Response", "y": "Count"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv = st.session_state.results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and DeepSeek API*")

if __name__ == "__main__":
    main()
