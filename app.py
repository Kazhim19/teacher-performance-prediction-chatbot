import streamlit as st
import joblib
import json
import requests
import re
import os
from dotenv import load_dotenv
from preprocessing_function import PreprocessingFunction
import base64

# Load environment variables
load_dotenv()

class TeacherPerformanceChatbot:
    def __init__(self):
        self.load_model_and_features()
        self.load_env_config()
        self.setup_llm_config()
        
    def load_model_and_features(self):
        """Load the trained model and feature columns"""
        try:
            self.model = joblib.load('teacher_performance_model.pkl')
            with open('feature_columns.json', 'r') as f:
                self.feature_columns = json.load(f)
            with open('raw_features.json', 'r') as f:
                self.raw_features = json.load(f)
            
            # Load model metadata for AUC display
            try:
                with open('model_metadata.json', 'r') as f:
                    self.model_metadata = json.load(f)
            except:
                self.model_metadata = {"test_auc": "N/A", "cv_auc": "N/A"}
            
        except Exception as e:
            self.model_metadata = {"test_auc": "N/A", "cv_auc": "N/A"}
            
    def load_env_config(self):
        """Load configuration from environment variables"""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4.1')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        
    def setup_llm_config(self):
        """Setup LLM configuration options"""
        self.llm_options = {
            "OpenAI": "openai",
            "Ollama": "ollama"
        }
        
    def call_openai_api(self, prompt):
        """Call OpenAI API using environment variables"""
        if not self.openai_api_key:
            return "Error: OpenAI API key not found in environment variables"
            
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"
    
    def call_ollama_api(self, prompt):
        """Call Ollama local API using environment variables"""
        data = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=data,
                timeout=180
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error calling Ollama API: {str(e)}"
    
    def classify_intent_with_llm(self, user_input, llm_choice):
        """Use LLM to classify user intent"""
        prompt = f"""
        You are an intent classification system for EduBot, a teacher performance prediction chatbot.
        
        Analyze the user's input and classify it into one of these categories:
        
        1. **greeting** - User is greeting, saying hello, or starting a conversation
           Examples: "Hello", "Hi", "Good morning", "Hey there", "What's up"
        
        2. **identity** - User is asking about the chatbot's identity, purpose, or capabilities
           Examples: "Who are you?", "What do you do?", "Tell me about yourself", "What can you help with?"
        
        3. **prediction** - User wants teacher performance prediction or provides teacher data
           Examples: "Predict teacher performance", "I'm a math teacher with 5 years experience", "Can you analyze this teacher's data?"
        
        4. **out_of_scope** - User is asking about topics unrelated to teacher performance prediction
           Examples: "What's the weather?", "Tell me a joke", "How to cook pasta?", "What's 2+2?"
        
        User Input: "{user_input}"
        
        IMPORTANT: Respond with ONLY the category name (greeting, identity, prediction, or out_of_scope) and nothing else.
        """
        
        if llm_choice == "openai":
            response = self.call_openai_api(prompt)
        elif llm_choice == "ollama":
            response = self.call_ollama_api(prompt)
        else:
            return "prediction"  # Default fallback
        
        # Check for API errors
        if response.startswith("Error"):
            return "prediction"  # Default fallback
        
        # Extract the intent from response
        intent = response.strip().lower()
        
        # Validate the intent
        valid_intents = ['greeting', 'identity', 'prediction', 'out_of_scope']
        if intent in valid_intents:
            return intent
        else:
            # Try to find valid intent in the response
            for valid_intent in valid_intents:
                if valid_intent in intent:
                    return valid_intent
            return "prediction"  # Default fallback
    
    def generate_greeting_response(self, user_input, llm_choice):
        """Generate a friendly greeting response"""
        prompt = f"""
        You are EduBot, a friendly AI assistant specialized in teacher performance prediction.
        
        The user just greeted you: "{user_input}"
        
        Respond with a warm, professional greeting that:
        1. Greets them back appropriately
        2. Briefly introduces yourself as EduBot, a teacher performance prediction assistant
        3. Offers to help them with teacher performance analysis
        4. Keeps it concise and friendly
        
        Example tone: "Hello! I'm EduBot, your AI assistant for teacher performance prediction and analysis. I'm here to help you assess teaching effectiveness and provide insights for educational support. How can I assist you today?"
        """
        
        if llm_choice == "openai":
            return self.call_openai_api(prompt)
        elif llm_choice == "ollama":
            return self.call_ollama_api(prompt)
    
    def generate_identity_response(self, user_input, llm_choice):
        """Generate a response explaining the chatbot's identity and purpose"""
        prompt = f"""
        You are EduBot, an AI assistant specialized in teacher performance prediction.
        
        The user is asking about who you are: "{user_input}"
        
        Provide a comprehensive but concise response that explains:
        1. Your name (EduBot) and role as an AI assistant
        2. Your specialty in teacher performance prediction and analysis
        3. What you can help with (analyzing teacher data, predicting performance risks, providing recommendations)
        4. The types of information you work with (age, experience, education, performance scores, etc.)
        5. How you can support educational institutions in teacher development
        
        Keep it professional, informative, and engaging. Mention that you're here to help with early intervention and teacher support.
        """
        
        if llm_choice == "openai":
            return self.call_openai_api(prompt)
        elif llm_choice == "ollama":
            return self.call_ollama_api(prompt)
    
    def generate_out_of_scope_response(self, user_input, llm_choice):
        """Generate a response for out-of-scope queries"""
        prompt = f"""
        You are EduBot, an AI assistant specialized in teacher performance prediction.
        
        The user asked something outside your expertise: "{user_input}"
        
        Respond politely by:
        1. Acknowledging their question politely
        2. Explaining that you specialize specifically in teacher performance prediction and analysis
        3. Redirecting them to your main capabilities (teacher performance assessment, risk prediction, educational support recommendations)
        4. Inviting them to ask about teacher-related topics or share teacher information for analysis
        5. Being helpful and friendly while staying focused on your domain
        
        Keep it brief, polite, and redirect them back to your core function. Don't be dismissive - be understanding but clear about your specialization.
        """
        
        if llm_choice == "openai":
            return self.call_openai_api(prompt)
        elif llm_choice == "ollama":
            return self.call_ollama_api(prompt)
    
    def create_extraction_prompt(self, user_input):
        """Create prompt for extracting structured data from user input"""
        raw_features_str = ', '.join(self.raw_features)
        
        prompt = f"""
        You are a data extraction assistant for a teacher performance prediction system.
        
        Analyze the user's input and determine if it contains teacher performance data for prediction.
        
        User Input: "{user_input}"
        
        IMPORTANT: Only attempt data extraction if the user input contains specific teacher information for performance prediction.
        
        If the input contains teacher data, extract it as JSON with these features:
        {raw_features_str}
        
        Guidelines for extraction:
        - teacher_id: Can be null for prediction
        - age: Extract numeric age
        - gender: "M" or "F"
        - education_level: "Bachelors", "Masters", or "PhD"
        - years_exp: Numeric years of experience
        - employment_type: "Full-time" or "Part-time"
        - subject: "Math", "Science", "English", "History", "Physics", "Chemistry", "Biology"
        - perf_score: Performance score (0-100 scale)
        - student_outcomes: Student outcome score (0-100 scale)
        - peer_reviews: Peer review score (0-100 scale) - convert ratings like "3.9 out of 5" to percentage
        - attendance_rate: Attendance percentage (0-100)
        - admin_support: "Yes" or "No"
        - workload: Numeric workload indicator
        - resource_availability: "Low", "Medium", or "High"
        - date_of_hire: MM/DD/YYYY format
        - date_of_last_eval: MM/DD/YYYY format (default to today if not provided)
        - time_to_event: Can be null for prediction
        
        Return format:
        {{
            "has_teacher_data": true/false,
            "data": {{
                // teacher data if has_teacher_data is true
            }},
            "missing_fields": []
        }}
        
        If the input does not contain teacher performance data (e.g., greetings, general questions, off-topic), return:
        {{
            "has_teacher_data": false,
            "data": null,
            "missing_fields": []
        }}
        
        RETURN ONLY VALID JSON WITH NO COMMENTS OR ADDITIONAL TEXT.
        """
        return prompt
    
    def create_response_prompt(self, prediction_prob, risk_level, user_input):
        """Create prompt for generating human-readable response"""
        prompt = f"""
        You are EduBot, a helpful AI assistant for teacher performance prediction.
        
        A teacher's performance has been analyzed with the following results:
        - Prediction Probability: {prediction_prob:.3f} ({prediction_prob*100:.1f}%)
        - Risk Level: {risk_level}
        
        Original user query: "{user_input}"
        
        Provide a helpful, professional response that:
        1. Explains the prediction in simple terms
        2. Interprets the risk level
        3. Offers specific, actionable recommendations based on the risk level
        4. Maintains a supportive, non-judgmental tone
        5. Addresses the user directly and professionally
        
        Risk Level Guidelines:
        - Low Risk (0-25%): Positive reinforcement, maintain current practices
        - Medium Risk (25-50%): Some areas for improvement, proactive support
        - High Risk (50-75%): Significant concern, immediate intervention needed
        - Critical Risk (75%+): Urgent attention required, comprehensive support plan
        
        Keep response concise but informative (2-3 paragraphs). Start with a friendly acknowledgment.
        """
        return prompt
    
    def extract_json_from_response(self, response):
        """Extract JSON from LLM response - handles comments and formatting"""
        try:
            # Method 1: Try to parse response as-is first
            try:
                return json.loads(response.strip())
            except:
                pass
            
            # Method 2: Remove comments and try again
            cleaned_response = response
            # Remove // style comments
            cleaned_response = re.sub(r'//.*?(?=\n|$)', '', cleaned_response)
            # Remove /* */ style comments  
            cleaned_response = re.sub(r'/\*.*?\*/', '', cleaned_response, flags=re.DOTALL)
            
            try:
                return json.loads(cleaned_response.strip())
            except:
                pass
                
            # Method 3: Find JSON block within response
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            print(f"Could not parse JSON from response: {response}")
            return None
            
        except Exception as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response}")
            return None
    
    def predict_performance(self, teacher_data):
        """Make prediction using the trained model"""
        try:
            # Preprocess the data
            processed_data = PreprocessingFunction.preprocess_for_prediction(
                teacher_data, self.feature_columns
            )
            
            # Make prediction
            prediction_prob = self.model.predict_proba(processed_data)[0][1]
            
            # Determine risk level
            if prediction_prob < 0.25:
                risk_level = "Low Risk"
            elif prediction_prob < 0.50:
                risk_level = "Medium Risk"
            elif prediction_prob < 0.75:
                risk_level = "High Risk"
            else:
                risk_level = "Critical Risk"
            
            return prediction_prob, risk_level
            
        except Exception as e:
            return None, f"Error in prediction: {str(e)}"
    
    def process_user_input(self, user_input, llm_choice):
        """Main function to process user input through the complete pipeline"""
        
        # Step 1: Use LLM to classify the intent
        intent = self.classify_intent_with_llm(user_input, llm_choice)
        
        # Handle different intents
        if intent == 'greeting':
            response = self.generate_greeting_response(user_input, llm_choice)
            return response
        
        elif intent == 'identity':
            response = self.generate_identity_response(user_input, llm_choice)
            return response
        
        elif intent == 'out_of_scope':
            response = self.generate_out_of_scope_response(user_input, llm_choice)
            return response
        
        elif intent == 'prediction':
            # Step 2: Try to extract structured data using LLM
            extraction_prompt = self.create_extraction_prompt(user_input)
            
            if llm_choice == "openai":
                llm_response = self.call_openai_api(extraction_prompt)
            elif llm_choice == "ollama":
                llm_response = self.call_ollama_api(extraction_prompt)
            else:
                return "Invalid LLM choice"
            
            # Check for API errors
            if llm_response.startswith("Error"):
                return llm_response
            
            # Step 3: Parse JSON response
            extracted_data = self.extract_json_from_response(llm_response)
            
            if not extracted_data:
                return f"Could not process the request. Please try again or provide more specific teacher information."
            
            # Step 4: Check if teacher data was found
            if not extracted_data.get('has_teacher_data', False):
                # If LLM classified as prediction but no data found, ask for more info
                return "I understand you're interested in teacher performance prediction, but I need specific teacher information to analyze. Please provide details like age, experience, education level, subject taught, performance scores, etc."
            
            # Step 5: Check for missing fields
            if extracted_data.get('missing_fields'):
                missing_fields_msg = f"I found some teacher information, but I need more details to make a prediction. Missing: {', '.join(extracted_data['missing_fields'])}. Please provide these details so I can help you better."
                return missing_fields_msg
            
            # Step 6: Make ML prediction
            teacher_data = extracted_data['data']
            prediction_prob, risk_level = self.predict_performance(teacher_data)
            
            if prediction_prob is None:
                return f"I encountered an issue with the prediction: {risk_level}. Please check your data and try again."
            
            # Step 7: Generate human-readable response
            response_prompt = self.create_response_prompt(prediction_prob, risk_level, user_input)
            
            if llm_choice == "openai":
                final_response = self.call_openai_api(response_prompt)
            elif llm_choice == "ollama":
                final_response = self.call_ollama_api(response_prompt)
            
            return {
                'prediction_probability': prediction_prob,
                'risk_level': risk_level,
                'response': final_response,
                'extracted_data': teacher_data
            }
        
        # Fallback
        return "I'm not sure how to help with that. I specialize in teacher performance prediction. Please share teacher information for analysis, or ask me about what I can do!"

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_image("./robot.png")

def main():
    st.set_page_config(
        page_title="EduBot - Teacher Performance Prediction Assistant",
        page_icon="./robot.png",
        layout="wide"
    )
    
    st.markdown(
        f"""
        <style>
            .header-container {{
                display: flex;
                align-items: center;
                justify-content: flex-start;
                gap: 12px;
            }}
            .header-container img {{
                width: 50px;
            }}
            @media (max-width: 768px) {{
                .header-container {{
                    flex-direction: column;
                    text-align: center;
                }}
                .header-container img {{
                    width: 25%;
                }}
            }}
        </style>

        <div class="header-container">
            <img src="data:image/png;base64,{img_base64}" alt="EduBot">
            <h1>EduBot - Teacher Performance Prediction Assistant</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("*AI-powered early intervention system for teacher support*")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = TeacherPerformanceChatbot()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize processing state
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # LLM Selection
    llm_choice = st.sidebar.selectbox(
        "Choose LLM Provider:",
        list(st.session_state.chatbot.llm_options.keys())
    )
    
    # Display current configuration
    st.sidebar.subheader("📋 Current Settings")
    if llm_choice == "OpenAI":
        if st.session_state.chatbot.openai_api_key:
            st.sidebar.success("✅ OpenAI API Key loaded")
            st.sidebar.info(f"Model: {st.session_state.chatbot.openai_model}")
        else:
            st.sidebar.error("❌ OpenAI API Key not found in .env")
    elif llm_choice == "Ollama":
        st.sidebar.info(f"Model: {st.session_state.chatbot.ollama_model}")
        st.sidebar.info(f"URL: {st.session_state.chatbot.ollama_url}")
    
    # Model Performance Metrics
    st.sidebar.subheader("📊 Model Performance")
    try:
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            st.metric("Test AUC", f"{st.session_state.chatbot.model_metadata.get('test_auc', 'N/A'):.3f}")
        with col_b:
            st.metric("CV AUC", f"{st.session_state.chatbot.model_metadata.get('cv_auc', 'N/A'):.3f}")
    except:
        st.sidebar.info("Model metrics not available")
    
    # Clear chat history
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat with EduBot")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (user_msg, bot_response) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"**👤 You:** {user_msg}")
                
                # Bot response
                if isinstance(bot_response, dict):
                    st.markdown(f"**🤖 EduBot:** {bot_response['response']}")
                    
                    # Show prediction details in expander
                    with st.expander(f"📊 Prediction Details #{i+1}"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Risk Probability", f"{bot_response['prediction_probability']:.1%}")
                            st.metric("Risk Level", bot_response['risk_level'])
                        with col_b:
                            st.json(bot_response['extracted_data'])
                else:
                    st.markdown(f"**🤖 EduBot:** {bot_response}")
                
                st.markdown("---")
        
        # Chat input form with Ctrl+Enter support
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask me anything or provide teacher information for analysis:",
                height=100,
                placeholder="Try: 'Hello!', 'Who are you?', or 'I'm a 35-year-old math teacher with 8 years experience...'" if not st.session_state.is_processing else "Please wait for EduBot to finish processing...",
                disabled=st.session_state.is_processing
            )
            
            submitted = st.form_submit_button(
                "➤ Send" if not st.session_state.is_processing else "🧠 Processing...", 
                type="primary",
                disabled=st.session_state.is_processing
            )
            
            if submitted and user_input.strip() and not st.session_state.is_processing:
                # Check if configuration is valid
                if llm_choice == "OpenAI" and not st.session_state.chatbot.openai_api_key:
                    st.error("❌ OpenAI API key not found in .env file")
                else:
                    # Set processing state and store the input
                    st.session_state.is_processing = True
                    st.session_state.pending_input = user_input
                    st.rerun()
            elif submitted and not user_input.strip():
                st.warning("Please enter a message")
        
        # Process the request when in processing state
        if st.session_state.is_processing and 'pending_input' in st.session_state:
            current_input = st.session_state.pending_input
            
            with st.spinner("🧠 EduBot is thinking..."):
                llm_type = st.session_state.chatbot.llm_options[llm_choice]
                
                # Process the input
                result = st.session_state.chatbot.process_user_input(
                    current_input, llm_type
                )
                
                # Add to chat history
                st.session_state.chat_history.append((current_input, result))
                
                # Reset processing state
                st.session_state.is_processing = False
                del st.session_state.pending_input
                
            st.rerun()
    
    with col2:
        st.subheader("📝 Required Information")
        st.markdown("""
        **Basic Info:**
        - Age
        - Gender (M/F)
        - Years of experience
        - Education level
        
        **Employment:**
        - Employment type
        - Subject taught
        - Date of hire
        
        **Performance:**
        - Performance score
        - Student outcomes
        - Peer reviews
        - Attendance rate
        
        **Support:**
        - Admin support (Yes/No)
        - Workload level
        - Resource availability
        """)
        
        st.subheader("🎯 Risk Levels")
        st.markdown("""
        - **🟢 Low (0-25%)**: Excellent
        - **🟡 Medium (25-50%)**: Monitor
        - **🟠 High (50-75%)**: Intervention
        - **🔴 Critical (75%+)**: Urgent Action
        """)

if __name__ == "__main__":
    main()
