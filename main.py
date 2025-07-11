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
        """Parse questionnaire text and extract questions with improved detection"""
        questions = []
        
        # Store original text for debugging
        original_text = text
        
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
            r'1.*never.*7.*always',
            r'please.*rate.*each.*statement'
        ]
        
        # Multiple strategies to find questions
        all_questions = []
        
        # Strategy 1: Find numbered questions (1., 2., 3., etc.)
        numbered_patterns = [
            r'(?:^|\n)\s*(\d+)\.\s+([^\n]+(?:\n(?!\s*\d+\.)[^\n]*)*)',
            r'(?:^|\n)\s*(\d+)\s*\.\s*([^\d]+?)(?=\n\s*\d+\.|$)',
            r'(\d+)\.\s*([^.]{10,}?)(?=\n|\d+\.|$)'
        ]
        
        for pattern in numbered_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    num = int(match.group(1))
                    question_text = match.group(2).strip()
                    if len(question_text) > 10 and not any('section' in q.lower() for q in [question_text]):
                        all_questions.append((num, f"Q{num}", question_text, match.start(), "numbered"))
                except:
                    continue
        
        # Strategy 2: Find Q-format questions (Q1:, Q2:, etc.)
        q_patterns = [
            r'(?:^|\n)\s*(Q\d+)[:.]?\s*([^\n]+(?:\n(?!Q\d+)[^\n]*)*)',
            r'(Q\d+)[:.]?\s*([^Q]{10,}?)(?=Q\d+|$)'
        ]
        
        for pattern in q_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    q_id = match.group(1)
                    question_text = match.group(2).strip()
                    num_search = re.search(r'\d+', q_id)
                    if num_search and len(question_text) > 10:
                        num = int(num_search.group())
                        all_questions.append((num, q_id, question_text, match.start(), "q_format"))
                except:
                    continue
        
        # Strategy 3: Find "Question X:" format
        question_patterns = [
            r'(?:^|\n)\s*Question\s+(\d+)[:.]?\s*([^\n]+(?:\n(?!Question\s+\d+)[^\n]*)*)',
            r'Question\s+(\d+)[:.]?\s*([^Q]{10,}?)(?=Question\s+\d+|$)'
        ]
        
        for pattern in question_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    num = int(match.group(1))
                    question_text = match.group(2).strip()
                    if len(question_text) > 10:
                        all_questions.append((num, f"Q{num}", question_text, match.start(), "question_format"))
                except:
                    continue
        
        # Strategy 4: Find questions ending with "?" 
        question_mark_pattern = r'([A-Z][^?]{20,}?\?)'
        question_mark_matches = re.finditer(question_mark_pattern, text)
        question_counter = 1000  # High number to avoid conflicts
        
        for match in question_mark_matches:
            question_text = match.group(1).strip()
            if len(question_text) > 20 and question_text.count('?') == 1:
                all_questions.append((question_counter, f"Q{question_counter}", question_text, match.start(), "question_mark"))
                question_counter += 1
        
        # Remove duplicates and sort
        seen_texts = set()
        unique_questions = []
        
        for num, q_id, text_content, pos, method in all_questions:
            # Clean text for comparison
            clean_text = re.sub(r'\s+', ' ', text_content.lower()).strip()
            if clean_text not in seen_texts and len(clean_text) > 10:
                seen_texts.add(clean_text)
                unique_questions.append((num, q_id, text_content, pos, method))
        
        # Sort by position in original text
        unique_questions.sort(key=lambda x: x[3])
        
        # Process each question to determine type and options
        for i, (num, q_id, question_text, start_pos, method) in enumerate(unique_questions):
            # Get surrounding context
            context_start = max(0, start_pos - 300)
            context_end = min(len(text), start_pos + len(question_text) + 300)
            context = text[context_start:context_end].lower()
            
            # Default values
            question_type = "open_ended"
            options = []
            
            # Check for multiple choice patterns
            mc_patterns = [
                r'[a-h]\)\s*([^a-h\)]{3,50}?)(?=[a-h]\)|$)',
                r'[a-h]\.\s*([^a-h\.]{3,50}?)(?=[a-h]\.|$)',
                r'[a-h]\s*\)\s*([^a-h\)]{3,50}?)(?=[a-h]\s*\)|$)'
            ]
            
            for pattern in mc_patterns:
                choice_matches = re.findall(pattern, question_text, re.IGNORECASE)
                if len(choice_matches) >= 2:
                    question_type = "multiple_choice"
                    options = [opt.strip() for opt in choice_matches if opt.strip()]
                    # Clean question text
                    question_text = re.split(r'[a-h][\)\.]', question_text, 1)[0].strip()
                    break
            
            # Check for Likert scale indicators
            if question_type != "multiple_choice":
                for pattern in likert_indicators:
                    if re.search(pattern, context, re.IGNORECASE):
                        question_type = "likert"
                        if re.search(r'1.*?7|scale.*?7', context):
                            options = ["1", "2", "3", "4", "5", "6", "7"]
                        else:
                            options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
                        break
            
            # Final cleanup and validation
            question_text = question_text.strip()
            if len(question_text) >= 10 and not any(skip in question_text.lower() for skip in ['section', 'instructions', 'please answer']):
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
                    
                    # Show extracted text (first 1000 chars for better debugging)
                    with st.expander("View Extracted Text (Preview)"):
                        st.text_area("Extracted PDF Text", pdf_text[:1000], height=200, disabled=True)
                        if len(pdf_text) > 1000:
                            st.info(f"Showing first 1000 characters of {len(pdf_text)} total characters")
                    
                    # Parse questionnaire
                    questions = PDFProcessor.parse_questionnaire(pdf_text)
                    st.session_state.questions = questions
                    
                    if len(questions) > 0:
                        st.success(f"Found {len(questions)} questions in the questionnaire")
                    else:
                        st.warning("No questions detected automatically. You can add them manually below.")
                    
                    # Manual question addition option
                    st.subheader("Manual Question Management")
                    
                    with st.expander("➕ Add Questions Manually"):
                        st.info("If automatic detection didn't work, you can add questions manually here.")
                        
                        with st.form("add_question_form"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                manual_text = st.text_area("Question Text", placeholder="Enter your question here...")
                            
                            with col2:
                                manual_type = st.selectbox("Question Type", ["open_ended", "likert", "multiple_choice"])
                                
                                if manual_type == "likert":
                                    scale_type = st.selectbox("Scale Type", ["5-point", "7-point"])
                                elif manual_type == "multiple_choice":
                                    manual_options = st.text_area("Options (one per line)", placeholder="Option 1\nOption 2\nOption 3")
                            
                            if st.form_submit_button("Add Question"):
                                if manual_text.strip():
                                    # Determine options based on type
                                    if manual_type == "likert":
                                        if scale_type == "7-point":
                                            options = ["1", "2", "3", "4", "5", "6", "7"]
                                        else:
                                            options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
                                    elif manual_type == "multiple_choice":
                                        options = [opt.strip() for opt in manual_options.split('\n') if opt.strip()]
                                    else:
                                        options = []
                                    
                                    # Add to session state
                                    new_question = Question(
                                        id=f"Q{len(st.session_state.questions) + 1}",
                                        text=manual_text.strip(),
                                        type=manual_type,
                                        options=options
                                    )
                                    st.session_state.questions.append(new_question)
                                    st.success("Question added successfully!")
                                    st.rerun()
                    
                    # Show current questions
                    if st.session_state.questions:
                        st.subheader("Current Questions")
                        
                        for i, question in enumerate(st.session_state.questions):
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
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.write(f"**Text:** {question.text}")
                                    st.write(f"**Type:** {question.type}")
                                    if question.options:
                                        st.write(f"**Options:** {', '.join(question.options)}")
                                    if issues:
                                        st.warning(f"Issues: {', '.join(issues)}")
                                
                                with col2:
                                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                                        st.session_state.questions.pop(i)
                                        st.rerun()
                        
                        # Clear all questions button
                        if st.button("🗑️ Clear All Questions", type="secondary"):
                            st.session_state.questions = []
                            st.rerun()
                        
                        # Summary
                        likert_count = sum(1 for q in st.session_state.questions if q.type == "likert")
                        mc_count = sum(1 for q in st.session_state.questions if q.type == "multiple_choice")
                        open_count = len(st.session_state.questions) - likert_count - mc_count
                        
                        st.info(f"""
                        **Summary:** {len(st.session_state.questions)} questions total
                        - {likert_count} Likert scale questions
                        - {mc_count} Multiple choice questions  
                        - {open_count} Open-ended questions
                        """)
                    
                    else:
                        st.info("No questions found. Please use the manual addition tool above to add your questions.")

    
    with col2:
        st.header("🎯 Simulation Results")
        
        # Always show simulation section if we have questions
        if st.session_state.questions and len(st.session_state.questions) > 0:
            valid_questions = [q for q in st.session_state.questions if len(q.text.strip()) >= 10]
            
            # Show simulation status
            if len(valid_questions) == 0:
                st.error("❌ No valid questions found. Please check your questions.")
                st.info("💡 Questions need at least 10 characters to be considered valid.")
            else:
                st.success(f"✅ Ready to simulate with {len(valid_questions)} valid questions")
            
            # Always show simulation controls (even if no valid questions, for debugging)
            st.subheader("🚀 Run Simulation")
            
            if len(valid_questions) > 0:
                # Test run option
                col_test, col_full = st.columns(2)
                
                with col_test:
                    if st.button("🧪 Test Run\n(3 participants)", type="secondary", use_container_width=True, key="test_run_btn"):
                        simulator = ExperimentSimulator(api_key)
                        
                        with st.spinner("Running test simulation..."):
                            test_results = simulator.simulate_responses(valid_questions, sample_chars, 3)
                            st.session_state.test_results = test_results
                        
                        st.success("Test completed!")
                        
                        # Show test results preview
                        with st.expander("📊 Test Results Preview"):
                            st.dataframe(test_results, use_container_width=True)
                        
                        # Show API usage
                        usage = simulator.deepseek_api.get_usage_stats()
                        st.info(f"**API Usage:** {usage['requests']} requests, {usage['total_tokens']} tokens")
                
                with col_full:
                    if st.button("🚀 Full Simulation", type="primary", use_container_width=True, key="full_sim_btn"):
                        # Show estimation in a container that doesn't affect button visibility
                        estimation_container = st.container()
                        
                        # Confirmation checkbox
                        confirm_simulation = st.checkbox(
                            "✅ I understand this will use my DeepSeek API quota and want to proceed",
                            key="confirm_full_sim"
                        )
                        
                        # Show estimation details
                        with estimation_container:
                            estimated_requests = len(valid_questions) * num_participants
                            estimated_time = estimated_requests * 2  # 2 seconds per request estimate
                            
                            st.info(f"""
                            **📋 Simulation Details:**
                            - Questions: {len(valid_questions)}
                            - Participants: {num_participants}
                            - Estimated API requests: {estimated_requests}
                            - Estimated time: ~{estimated_time//60}m {estimated_time%60}s
                            """)
                        
                        if confirm_simulation:
                            if st.button("▶️ Start Full Simulation", type="primary", key="start_full_sim"):
                                simulator = ExperimentSimulator(api_key)
                                
                                with st.spinner("Running full simulation... Please wait."):
                                    start_time = time.time()
                                    
                                    # Add a more detailed progress indicator
                                    progress_container = st.container()
                                    with progress_container:
                                        st.info("🤖 AI participants are responding to your questionnaire...")
                                    
                                    results = simulator.simulate_responses(valid_questions, sample_chars, num_participants)
                                    end_time = time.time()
                                    st.session_state.results = results
                                
                                # Show completion stats
                                duration = end_time - start_time
                                usage = simulator.deepseek_api.get_usage_stats()
                                
                                st.success(f"""
                                🎉 **Simulation Completed Successfully!**
                                
                                **Performance Stats:**
                                - Duration: {duration//60:.0f}m {duration%60:.0f}s
                                - API Requests: {usage['requests']}
                                - Total Tokens: {usage['total_tokens']}
                                - Participants: {num_participants}
                                - Questions: {len(valid_questions)}
                                """)
                                
                                # Auto-scroll to results
                                st.balloons()
            else:
                st.warning("⚠️ Please ensure you have valid questions before running simulation.")
                st.info("💡 Add questions manually using the form above if automatic detection didn't work.")
        
        else:
            # No questions yet
            if uploaded_file is not None:
                st.warning("⚠️ No questions detected in PDF. Please check the format or add questions manually above.")
            else:
                st.info("📤 Please upload a PDF questionnaire to begin simulation.")
        
        # Results section (always show if results exist)
        if st.session_state.results is not None:
            st.markdown("---")
            st.subheader("📊 Simulation Results")
            
            # Results summary
            st.success(f"✅ Simulation complete! {len(st.session_state.results)} participants responded.")
            
            # Raw data viewer
            with st.expander("📋 Raw Data Table", expanded=False):
                st.dataframe(st.session_state.results, use_container_width=True)
            
            # Analysis section
            st.subheader("📈 Response Analysis")
            
            # Get Likert questions for analysis
            likert_questions = [q for q in st.session_state.questions if q.type == "likert"]
            
            if likert_questions:
                # Question selector
                selected_question = st.selectbox(
                    "🎯 Select a Likert Scale Question for Analysis:",
                    [f"{q.id}: {q.text[:50]}..." if len(q.text) > 50 else f"{q.id}: {q.text}" for q in likert_questions],
                    key="analysis_question_select"
                )
                
                if selected_question:
                    # Extract question ID
                    q_id = selected_question.split(":")[0]
                    
                    if q_id in st.session_state.results.columns:
                        question_data = st.session_state.results[q_id]
                        value_counts = question_data.value_counts()
                        
                        # Create visualization
                        col_chart, col_stats = st.columns([2, 1])
                        
                        with col_chart:
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"Response Distribution for {q_id}",
                                labels={"x": "Response", "y": "Count"},
                                color=value_counts.values,
                                color_continuous_scale="viridis"
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_stats:
                            st.metric("Total Responses", len(question_data))
                            st.metric("Unique Responses", len(value_counts))
                            
                            # Most common response
                            most_common = value_counts.index[0]
                            most_common_count = value_counts.iloc[0]
                            st.metric("Most Common", f"{most_common} ({most_common_count})")
            
            else:
                st.info("💡 No Likert scale questions found for detailed analysis.")
            
            # Export section
            st.subheader("📥 Export Results")
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                # CSV download
                csv_data = st.session_state.results.to_csv(index=False)
                st.download_button(
                    label="📊 Download as CSV",
                    data=csv_data,
                    file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_export2:
                # JSON download
                json_data = st.session_state.results.to_json(orient='records', indent=2)
                st.download_button(
                    label="📄 Download as JSON",
                    data=json_data,
                    file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Clear results option
            if st.button("🗑️ Clear Results", type="secondary"):
                st.session_state.results = None
                if 'test_results' in st.session_state:
                    del st.session_state.test_results
                st.rerun()

    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and DeepSeek API*")

if __name__ == "__main__":
    main()
