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
        
        # Split text into sections and find individual questions
        # Look for numbered questions more precisely
        question_matches = re.finditer(
            r'(?:^|\n)(?:Q?\s*)?(\d+)\.?\s*([^.]*?(?:\?|\.|\n(?=\d+\.)|$))', 
            text, 
            re.MULTILINE | re.DOTALL
        )
        
        # Also look for Q-format questions
        q_matches = re.finditer(
            r'(?:^|\n)(Q\d+)[:.]?\s*([^.]*?(?:\?|\.|\n(?=Q\d+)|$))', 
            text, 
            re.MULTILINE | re.DOTALL
        )
        
        # Combine and sort all matches
        all_matches = []
        for match in question_matches:
            num = int(match.group(1))
            question_text = match.group(2).strip()
            all_matches.append((num, f"Q{num}", question_text, match.start(), match.end()))
        
        for match in q_matches:
            q_id = match.group(1)
            question_text = match.group(2).strip()
            # Extract number from Q format
            num = int(re.search(r'\d+', q_id).group())
            all_matches.append((num, q_id, question_text, match.start(), match.end()))
        
        # Sort by position in text to maintain order
        all_matches.sort(key=lambda x: x[3])
        
        # Process each question
        for i, (num, q_id, question_text, start, end) in enumerate(all_matches):
            # Get context around the question for better type detection
            context_start = max(0, start - 200)
            context_end = min(len(text), end + 200)
            context = text[context_start:context_end].lower()
            
            # Determine question type
            question_type = "open_ended"
            options = []
            
            # Check for multiple choice options first
            choice_patterns = [
                r'[a-h]\)\s*([^a-h\)]*?)(?=[a-h]\)|$)',
                r'[a-h]\.?\s*([^a-h\.]*?)(?=[a-h]\.|$)'
            ]
            
            for pattern in choice_patterns:
                matches = re.findall(pattern, question_text.lower())
                if len(matches) >= 2:  # At least 2 options found
                    question_type = "multiple_choice"
                    options = [opt.strip() for opt in matches if opt.strip()]
                    break
            
            # Check for Likert scale indicators in context
            if question_type != "multiple_choice":
                for pattern in likert_indicators:
                    if re.search(pattern, context):
                        question_type = "likert"
                        # Check if it's a 7-point scale
                        if re.search(r'1.*?7|scale.*?7', context):
                            options = ["1 - Never", "2 - Rarely", "3 - Sometimes", 
                                     "4 - Often", "5 - Very Often", "6 - Almost Always", "7 - Always"]
                        else:
                            options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
                        break
            
            # Clean up question text
            question_text = re.sub(r'[a-h]\).*

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
        
        # Group questions by type for more efficient processing
        likert_questions = [q for q in questions if q.type == "likert"]
        mc_questions = [q for q in questions if q.type == "multiple_choice"]
        open_questions = [q for q in questions if q.type == "open_ended"]
        
        for i in range(num_participants):
            participant_responses = {"participant_id": i + 1}
            
            status_text.text(f"Simulating participant {i + 1}/{num_participants}...")
            
            # Generate base participant profile
            base_prompt = self.generate_participant_prompt(sample_chars, i + 1)
            
            # Process questions in batches by type for efficiency
            if likert_questions:
                likert_batch = self._process_likert_batch(base_prompt, likert_questions)
                participant_responses.update(likert_batch)
            
            if mc_questions:
                mc_batch = self._process_multiple_choice_batch(base_prompt, mc_questions)
                participant_responses.update(mc_batch)
            
            if open_questions:
                open_batch = self._process_open_ended_batch(base_prompt, open_questions)
                participant_responses.update(open_batch)
            
            results.append(participant_responses)
            progress_bar.progress((i + 1) / num_participants)
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(results)
    
    def _process_likert_batch(self, base_prompt: str, questions: List[Question]) -> Dict[str, str]:
        """Process Likert scale questions in batch"""
        if not questions:
            return {}
        
        # Create batch prompt for Likert questions
        likert_prompt = f"""
        {base_prompt}
        
        Please answer the following Likert scale questions. For each question, respond with EXACTLY one of these options:
        - Strongly Disagree
        - Disagree  
        - Neutral
        - Agree
        - Strongly Agree
        
        (For 7-point scales, use: 1, 2, 3, 4, 5, 6, 7)
        
        Questions:
        """
        
        for i, question in enumerate(questions[:5]):  # Batch first 5 questions
            likert_prompt += f"{question.id}: {question.text}\n"
        
        likert_prompt += "\nProvide your answers in this exact format:\nQ1: [Your Answer]\nQ2: [Your Answer]\n"
        
        response = self.deepseek_api.generate_response(likert_prompt, max_tokens=300)
        
        # Parse batch response
        responses = {}
        for question in questions[:5]:
            # Extract answer for this question
            match = re.search(f"{question.id}:\\s*([^\n]+)", response)
            if match:
                answer = match.group(1).strip()
                # Validate Likert response
                valid_responses = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree", "1", "2", "3", "4", "5", "6", "7"]
                if any(valid in answer for valid in valid_responses):
                    responses[question.id] = answer
                else:
                    responses[question.id] = "Neutral"  # Default fallback
            else:
                responses[question.id] = "Neutral"
        
        # Process remaining questions individually if more than 5
        if len(questions) > 5:
            for question in questions[5:]:
                single_prompt = f"""
                {base_prompt}
                
                Answer this Likert scale question with one of: {', '.join(question.options)}
                
                {question.id}: {question.text}
                
                Answer only:
                """
                response = self.deepseek_api.generate_response(single_prompt, max_tokens=50)
                responses[question.id] = response.strip()
        
        return responses
    
    def _process_multiple_choice_batch(self, base_prompt: str, questions: List[Question]) -> Dict[str, str]:
        """Process multiple choice questions in batch"""
        if not questions:
            return {}
        
        responses = {}
        for question in questions:
            if question.options:
                mc_prompt = f"""
                {base_prompt}
                
                Choose the best answer for this multiple choice question:
                
                {question.text}
                
                Options: {', '.join(question.options)}
                
                Respond with just the letter (a, b, c, etc.) or the exact option text:
                """
            else:
                mc_prompt = f"""
                {base_prompt}
                
                Answer this multiple choice question:
                {question.text}
                
                Provide a brief, direct answer:
                """
            
            response = self.deepseek_api.generate_response(mc_prompt, max_tokens=100)
            responses[question.id] = response.strip()
        
        return responses
    
    def _process_open_ended_batch(self, base_prompt: str, questions: List[Question]) -> Dict[str, str]:
        """Process open-ended questions individually"""
        if not questions:
            return {}
        
        responses = {}
        for question in questions:
            open_prompt = f"""
            {base_prompt}
            
            Answer this open-ended question in 1-3 sentences, staying in character:
            
            {question.text}
            
            Your response:
            """
            
            response = self.deepseek_api.generate_response(open_prompt, max_tokens=150)
            responses[question.id] = response.strip()
        
        return responses

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
                    
                    # Add option to manually edit questions
                    if st.checkbox("Enable Manual Question Editing"):
                        st.info("💡 You can edit question types and options if the automatic detection needs adjustment")
                        
                        edited_questions = []
                        for i, question in enumerate(questions):
                            with st.expander(f"Question {i+1}: {question.id} - {question.type}"):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    new_text = st.text_area(f"Question Text {i+1}", value=question.text, key=f"text_{i}")
                                
                                with col2:
                                    new_type = st.selectbox(
                                        f"Question Type {i+1}", 
                                        ["likert", "multiple_choice", "open_ended"],
                                        index=["likert", "multiple_choice", "open_ended"].index(question.type),
                                        key=f"type_{i}"
                                    )
                                
                                if new_type == "likert":
                                    scale_type = st.selectbox(
                                        f"Likert Scale Type {i+1}",
                                        ["5-point (Strongly Disagree to Strongly Agree)", "7-point (1 to 7)"],
                                        key=f"scale_{i}"
                                    )
                                    if "5-point" in scale_type:
                                        new_options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
                                    else:
                                        new_options = ["1", "2", "3", "4", "5", "6", "7"]
                                elif new_type == "multiple_choice":
                                    options_text = st.text_area(
                                        f"Options {i+1} (one per line)", 
                                        value='\n'.join(question.options) if question.options else "",
                                        key=f"options_{i}"
                                    )
                                    new_options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]
                                else:
                                    new_options = []
                                
                                edited_questions.append(Question(
                                    id=question.id,
                                    text=new_text,
                                    type=new_type,
                                    options=new_options
                                ))
                        
                        st.session_state.questions = edited_questions
                        if st.button("Save Edited Questions"):
                            st.success("Questions updated successfully!")
                            st.rerun()
                    
                    else:
                        # Show read-only questions with quality indicators
                        for i, question in enumerate(questions):
                            # Determine quality indicator
                            quality = "✅"  # Good
                            issues = []
                            
                            if question.type == "multiple_choice" and len(question.options) < 2:
                                quality = "⚠️"  # Warning
                                issues.append("Few or no options detected")
                            
                            if question.type == "likert" and not question.options:
                                quality = "⚠️"
                                issues.append("No scale options detected")
                            
                            if len(question.text) < 10:
                                quality = "❌"  # Error
                                issues.append("Question text too short")
                            
                            with st.expander(f"{quality} Question {i+1}: {question.id} ({question.type})"):
                                st.write(f"**Text:** {question.text}")
                                st.write(f"**Type:** {question.type}")
                                if question.options:
                                    st.write(f"**Options:** {', '.join(question.options)}")
                                if issues:
                                    st.warning(f"Issues: {', '.join(issues)}")
                        
                        # Quality summary
                        good_questions = sum(1 for q in questions if len(q.text) >= 10)
                        likert_detected = sum(1 for q in questions if q.type == "likert")
                        mc_detected = sum(1 for q in questions if q.type == "multiple_choice")
                        
                        st.info(f"""
                        **Parsing Summary:**
                        - {good_questions}/{len(questions)} questions parsed successfully
                        - {likert_detected} Likert scale questions detected
                        - {mc_detected} multiple choice questions detected
                        - {len(questions) - likert_detected - mc_detected} open-ended questions
                        
                        💡 Enable manual editing above if you need to fix any detection issues.
                        """)
    
    with col2:
        st.header("🎯 Simulation Results")
        
        if st.session_state.questions and len(st.session_state.questions) > 0:
            # Check if questions are valid
            valid_questions = [q for q in st.session_state.questions if len(q.text.strip()) >= 10]
            
            if len(valid_questions) == 0:
                st.error("❌ No valid questions found. Please check your PDF or use manual editing to fix question extraction.")
                st.stop()
            
            # Test simulation option
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧪 Test Run (3 participants)", type="secondary"):
                    simulator = ExperimentSimulator(api_key)
                    
                    with st.spinner("Running test simulation..."):
                        test_results = simulator.simulate_responses(
                            valid_questions,  # Use only valid questions
                            sample_chars,
                            3  # Test with 3 participants
                        )
                        st.session_state.test_results = test_results
                    
                    st.success("Test simulation completed!")
                    
                    # Show test results
                    with st.expander("Test Results Preview"):
                        st.dataframe(test_results)
                    
                    # Show API usage
                    usage = simulator.deepseek_api.get_usage_stats()
                    st.info(f"API Usage: {usage['requests']} requests, {usage['total_tokens']} tokens")
            
            with col2:
                if st.button("🚀 Run Full Simulation", type="primary"):
                    simulator = ExperimentSimulator(api_key)
                    
                    # Estimate time and cost
                    estimated_requests = len(valid_questions) * num_participants
                    estimated_time = estimated_requests * 2  # Rough estimate: 2 seconds per request
                    
                    st.warning(f"""
                    **Estimated simulation details:**
                    - Valid Questions: {len(valid_questions)}/{len(st.session_state.questions)}
                    - Participants: {num_participants}
                    - Estimated API requests: {estimated_requests}
                    - Estimated time: {estimated_time//60} minutes {estimated_time%60} seconds
                    
                    ⚠️ This will use your DeepSeek API quota. Consider running a test first.
                    """)
                    
                    if st.checkbox("I understand and want to proceed"):
                        with st.spinner("Running full simulation..."):
                            start_time = time.time()
                            results = simulator.simulate_responses(
                                valid_questions,  # Use only valid questions
                                sample_chars,
                                num_participants
                            )
                            end_time = time.time()
                            st.session_state.results = results
                        
                        # Show completion stats
                        duration = end_time - start_time
                        usage = simulator.deepseek_api.get_usage_stats()
                        
                        st.success(f"""
                        ✅ Simulation completed successfully!
                        
                        **Statistics:**
                        - Duration: {duration//60:.0f}m {duration%60:.0f}s
                        - API Requests: {usage['requests']}
                        - Total Tokens: {usage['total_tokens']}
                        - Participants: {num_participants}
                        - Questions: {len(valid_questions)}
                        """)
        
        elif uploaded_file is not None:
            st.warning("⚠️ No questions detected in the uploaded PDF. Please check the file format or try manual editing.")
        else:
            st.info("📤 Please upload a PDF questionnaire to begin.")


        
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
, '', question_text, flags=re.DOTALL)
            question_text = question_text.strip()
            
            # Skip if question text is too short or just section headers
            if len(question_text) < 10 or 'section' in question_text.lower():
                continue
            
            questions.append(Question(
                id=q_id,
                text=question_text,
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
