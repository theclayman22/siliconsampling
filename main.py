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
        self.request_count = 0
        self.total_tokens = 0
    
    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response using DeepSeek API with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=45  # Increased timeout
                )
                
                self.request_count += 1
                
                if response.status_code == 200:
                    result = response.json()
                    if "usage" in result:
                        self.total_tokens += result["usage"].get("total_tokens", 0)
                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        return f"Rate limit exceeded. Please try again later."
                else:
                    error_msg = f"API Error {response.status_code}"
                    try:
                        error_detail = response.json().get("error", {}).get("message", "")
                        if error_detail:
                            error_msg += f": {error_detail}"
                    except:
                        pass
                    return error_msg
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    return "Request timeout. Please try again."
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return f"Error: {str(e)}"
        
        return "Failed to get response after multiple attempts."
    
    def get_usage_stats(self) -> dict:
        """Get API usage statistics"""
        return {
            "requests": self.request_count,
            "total_tokens": self.total_tokens
        }

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
        
        # Clean and preprocess text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'-\s*', '', text)  # Remove word breaks
        
        # Enhanced Likert scale detection patterns
        likert_indicators = [
            r'strongly disagree.*?strongly agree',
            r'disagree.*?agree',
            r'never.*?always',
            r'scale.*?1.*?5',
            r'scale.*?1.*?7',
            r'1\s*=.*?5\s*=',
            r'1\s*=.*?7\s*=',
            r'indicate.*?level.*?agreement',
            r'rate.*?statement',
            r'1.*never.*7.*always'
        ]
        
        # Find numbered questions
        numbered_pattern = r'(?:^|\n)(\d+)\.\s*([^.\n]*(?:\.[^.\n]*)*?)(?=\n\d+\.|$)'
        numbered_matches = re.finditer(numbered_pattern, text, re.MULTILINE | re.DOTALL)
        
        # Find Q-format questions  
        q_pattern = r'(?:^|\n)(Q\d+)[:.]?\s*([^.\n]*(?:\.[^.\n]*)*?)(?=\n(?:Q\d+|$))'
        q_matches = re.finditer(q_pattern, text, re.MULTILINE | re.DOTALL)
        
        # Combine all matches
        all_matches = []
        
        for match in numbered_matches:
            num = int(match.group(1))
            question_text = match.group(2).strip()
            all_matches.append((num, f"Q{num}", question_text, match.start()))
        
        for match in q_matches:
            q_id = match.group(1)
            question_text = match.group(2).strip()
            num_search = re.search(r'\d+', q_id)
            if num_search:
                num = int(num_search.group())
                all_matches.append((num, q_id, question_text, match.start()))
        
        # Sort by position in text
        all_matches.sort(key=lambda x: x[3])
        
        # Process each question
        for num, q_id, question_text, start_pos in all_matches:
            # Get surrounding context for better type detection
            context_start = max(0, start_pos - 200)
            context_end = min(len(text), start_pos + len(question_text) + 200)
            context = text[context_start:context_end].lower()
            
            # Default values
            question_type = "open_ended"
            options = []
            
            # Check for multiple choice patterns
            choice_pattern = r'[a-h]\)\s*([^a-h\)]+?)(?=[a-h]\)|$)'
            choice_matches = re.findall(choice_pattern, question_text, re.IGNORECASE)
            
            if len(choice_matches) >= 2:
                question_type = "multiple_choice"
                options = [opt.strip() for opt in choice_matches if opt.strip()]
                # Clean question text by removing options
                clean_pattern = r'[a-h]\).*$'
                question_text = re.sub(clean_pattern, '', question_text, flags=re.DOTALL | re.IGNORECASE)
                question_text = question_text.strip()
            
            # Check for Likert scale indicators
            if question_type != "multiple_choice":
                for pattern in likert_indicators:
                    if re.search(pattern, context, re.IGNORECASE):
                        question_type = "likert"
                        # Determine scale type
                        if re.search(r'1.*?7|scale.*?7', context):
                            options = ["1", "2", "3", "4", "5", "6", "7"]
                        else:
                            options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
                        break
            
            # Skip if question is too short or appears to be a header
            if len(question_text.strip()) < 10 or 'section' in question_text.lower():
                continue
            
            questions.append(Question(
                id=q_id,
                text=question_text.strip(),
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
            
            status_text.text(f"Simulating participant {i + 1}/{num_participants}...")
            
            # Generate base participant profile
            base_prompt = self.generate_participant_prompt(sample_chars, i + 1)
            
            # Process each question
            for question in questions:
                question_prompt = f"""
                {base_prompt}
                
                Answer this question: {question.text}
                
                Type: {question.type}
                {f"Options: {', '.join(question.options)}" if question.options else ""}
                
                Provide only your answer:
                """
                
                response = self.deepseek_api.generate_response(question_prompt, max_tokens=150)
                participant_responses[question.id] = response.strip()
                
                # Small delay to avoid rate limiting
                time.sleep(0.2)
            
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
                    
                    # Display questions with quality indicators
                    st.subheader("Extracted Questions")
                    
                    for i, question in enumerate(questions):
                        # Quality indicator
                        quality = "✅"
                        issues = []
                        
                        if question.type == "multiple_choice" and len(question.options) < 2:
                            quality = "⚠️"
                            issues.append("Few options detected")
                        
                        if question.type == "likert" and not question.options:
                            quality = "⚠️"
                            issues.append("No scale options")
                        
                        if len(question.text) < 10:
                            quality = "❌"
                            issues.append("Text too short")
                        
                        with st.expander(f"{quality} Question {i+1}: {question.id} ({question.type})"):
                            st.write(f"**Text:** {question.text}")
                            st.write(f"**Type:** {question.type}")
                            if question.options:
                                st.write(f"**Options:** {', '.join(question.options)}")
                            if issues:
                                st.warning(f"Issues: {', '.join(issues)}")
                    
                    # Summary
                    likert_count = sum(1 for q in questions if q.type == "likert")
                    mc_count = sum(1 for q in questions if q.type == "multiple_choice")
                    open_count = len(questions) - likert_count - mc_count
                    
                    st.info(f"""
                    **Summary:** {len(questions)} questions total
                    - {likert_count} Likert scale questions
                    - {mc_count} Multiple choice questions  
                    - {open_count} Open-ended questions
                    """)
    
    with col2:
        st.header("🎯 Simulation Results")
        
        if st.session_state.questions and len(st.session_state.questions) > 0:
            valid_questions = [q for q in st.session_state.questions if len(q.text.strip()) >= 10]
            
            if len(valid_questions) == 0:
                st.error("❌ No valid questions found. Please check your PDF.")
                st.stop()
            
            # Test run option
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧪 Test Run (3 participants)", type="secondary"):
                    simulator = ExperimentSimulator(api_key)
                    
                    with st.spinner("Running test simulation..."):
                        test_results = simulator.simulate_responses(valid_questions, sample_chars, 3)
                        st.session_state.test_results = test_results
                    
                    st.success("Test completed!")
                    with st.expander("Test Results"):
                        st.dataframe(test_results)
                    
                    usage = simulator.deepseek_api.get_usage_stats()
                    st.info(f"API Usage: {usage['requests']} requests, {usage['total_tokens']} tokens")
            
            with col2:
                if st.button("🚀 Run Full Simulation", type="primary"):
                    simulator = ExperimentSimulator(api_key)
                    
                    estimated_requests = len(valid_questions) * num_participants
                    estimated_time = estimated_requests * 2
                    
                    st.warning(f"""
                    **Estimated simulation:**
                    - Questions: {len(valid_questions)}
                    - Participants: {num_participants}
                    - API requests: {estimated_requests}
                    - Time: ~{estimated_time//60}m {estimated_time%60}s
                    """)
                    
                    if st.checkbox("I understand and want to proceed"):
                        with st.spinner("Running simulation..."):
                            start_time = time.time()
                            results = simulator.simulate_responses(valid_questions, sample_chars, num_participants)
                            end_time = time.time()
                            st.session_state.results = results
                        
                        duration = end_time - start_time
                        usage = simulator.deepseek_api.get_usage_stats()
                        
                        st.success(f"""
                        ✅ Simulation completed!
                        - Duration: {duration//60:.0f}m {duration%60:.0f}s
                        - API Requests: {usage['requests']}
                        - Tokens: {usage['total_tokens']}
                        """)
        
        elif uploaded_file is not None:
            st.warning("⚠️ No questions detected in PDF. Please check the format.")
        else:
            st.info("📤 Please upload a PDF questionnaire to begin.")
        
        # Display results
        if st.session_state.results is not None:
            st.subheader("📊 Results")
            
            # Raw data
            with st.expander("Raw Data"):
                st.dataframe(st.session_state.results)
            
            # Analysis for Likert questions
            st.subheader("📈 Analysis")
            
            likert_questions = [q for q in st.session_state.questions if q.type == "likert"]
            
            if likert_questions:
                selected_question = st.selectbox(
                    "Select Likert Question for Analysis",
                    [q.id for q in likert_questions]
                )
                
                if selected_question and selected_question in st.session_state.results.columns:
                    question_data = st.session_state.results[selected_question]
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
