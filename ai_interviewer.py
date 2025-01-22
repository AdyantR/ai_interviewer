import spacy
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
from gtts import gTTS
import os
import speech_recognition as sr
import time

class AIInterviewGenerator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        self.question_generator = pipeline(
            "text2text-generation",
            model=self.model.to(self.device),  # Ensure the model is on the correct device
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.recognizer = sr.Recognizer()

    def extract_topics(self, text):
        """Extract topics and context using NLP."""
        doc = self.nlp(text)
        technical_contexts = []
        for sent in doc.sents:
            tech_terms = []
            context = []
            for token in sent:
                if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and len(token.text) > 2:
                    tech_terms.append(token.text.lower())
                if not token.is_stop and not token.is_punct:
                    context.append(token.text.lower())
            
            if tech_terms:
                technical_contexts.append({
                    'terms': tech_terms,
                    'context': ' '.join(context),
                    'full_sentence': sent.text
                })
        
        return technical_contexts

    def match_topics(self, job_topics, resume_topics):
        """Match relevant topics between job requirements and resume experience."""
        matched_topics = []
        used_combinations = set()
        
        for job_topic in job_topics:
            for resume_topic in resume_topics:
                common_terms = set(job_topic['terms']) & set(resume_topic['terms'])
                if common_terms:
                    term_key = tuple(sorted(list(common_terms)))
                    if term_key not in used_combinations:
                        matched_topics.append({
                            'terms': list(common_terms)[:2],
                            'job_context': job_topic['full_sentence'],
                            'resume_context': resume_topic['full_sentence']
                        })
                        used_combinations.add(term_key)
        
        return matched_topics

    def generate_question_prompt(self, topic):
        """Generate a dynamic prompt for question generation."""
        terms = ', '.join(topic['terms'])
        prompt = f"""
        Generate a technical interview question about {terms}.
        
        Context from job requirements:
        {topic['job_context']}
        
        Context from candidate experience:
        {topic['resume_context']}
        
        Requirements for the question:
        1. Must be specific to {terms}
        2. Should focus on system design, optimization, or problem-solving
        3. Must require detailed technical knowledge
        4. Should address real-world challenges
        5. Must be clear and direct
        
        The question should explore:
        - Technical implementation details
        - Performance considerations
        - Scalability aspects
        - Best practices
        """
        return prompt

    def generate_follow_up_prompt(self, original_question, candidate_response, topic):
        """Generate a contextual follow-up question based on the initial question and response."""
        return f"""
        Original question: {original_question}
        Candidate's response: {candidate_response}
        Technical focus: {', '.join(topic['terms'])}
        
        Generate a follow-up question that:
        1. Directly relates to the original question about {', '.join(topic['terms'])}
        2. Probes deeper into technical specifics mentioned in the response
        3. Asks about specific implementation details, trade-offs, or optimization strategies
        4. Maintains focus on the same technical topic but explores a different aspect
        5. Requires concrete examples or specific technical explanations
        
        The follow-up must:
        - Build upon the context of the original question
        - Be more specific than the first question
        - Not repeat any aspects already covered
        - End with a question mark
        """

    def clean_question(self, question):
        """Clean and format the generated question."""
        question = re.sub(r'[#*`]', '', question)
        # Remove common generic starts
        question = re.sub(r'^(Can you|Could you|Please|Tell me|Explain|Describe) ', '', question)
        if not question.strip().endswith('?'):
            question = question.strip() + '?'
        return question.strip()

    def text_to_speech(self, text):
        """Convert text to speech and play it."""
        try:
            tts = gTTS(text)
            tts.save("question.mp3")
            os.system("start question.mp3" if os.name == 'nt' else "open question.mp3")
            time.sleep(len(text.split()) / 2.5)
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            print("Question (text only):", text)

    def speech_to_text(self):
        """Capture verbal response via microphone."""
        with sr.Microphone() as source:
            print("\n[Listening for response...]\n")
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=30, phrase_time_limit=120)
                response = self.recognizer.recognize_google(audio)
                return response
            except sr.UnknownValueError:
                return "Could not understand or hear the response. Please try again."
            except sr.RequestError as e:
                return f"Speech recognition error: {e}"
            except Exception as e:
                return f"Error capturing response: {e}"

    def conduct_interview(self, job_post, resume, num_questions=3):
        """Conduct an interactive interview with audio."""
        print("\n=== Starting Interview ===\n")
        
        job_topics = self.extract_topics(job_post)
        resume_topics = self.extract_topics(resume)
        matched_topics = self.match_topics(job_topics, resume_topics)
        
        if not matched_topics:
            print("Error: Could not generate valid technical questions.")
            return
        
        used_questions = set()
        interview_log = []
        question_count = 0
        
        for topic in matched_topics:
            if question_count >= num_questions:
                break
                
            # Generate main question
            prompt = self.generate_question_prompt(topic)
            main_question = self.question_generator(
                prompt,
                max_length=150,
                num_beams=4,
                temperature=0.7,
                no_repeat_ngram_size=3
            )[0]['generated_text']
            
            main_question = self.clean_question(main_question)
            if main_question in used_questions:
                continue
                
            used_questions.add(main_question)
            question_count += 1
            
            print(f"\n--- Question {question_count} ---")
            print(f"Topic: {', '.join(topic['terms'])}")
            print(f"Q: {main_question}")
            
            # Ask question via audio
            self.text_to_speech(main_question)
            
            # Get response
            response = self.speech_to_text()
            print(f"\nCandidate's Response:\n{response}\n")
            
            # Generate follow-up
            follow_up_prompt = self.generate_follow_up_prompt(main_question, response, topic)
            follow_up = self.question_generator(
                follow_up_prompt,
                max_length=150,
                num_beams=4,
                temperature=0.7,
                no_repeat_ngram_size=3
            )[0]['generated_text']
            
            follow_up = self.clean_question(follow_up)
            if follow_up not in used_questions:
                used_questions.add(follow_up)
                print(f"Follow-up: {follow_up}")
                
                # Ask follow-up via audio
                self.text_to_speech(follow_up)
                
                # Get follow-up response
                follow_up_response = self.speech_to_text()
                print(f"\nCandidate's Follow-up Response:\n{follow_up_response}\n")
                
                interview_log.append({
                    'question_number': question_count,
                    'topic': topic['terms'],
                    'main_question': main_question,
                    'main_response': response,
                    'follow_up': follow_up,
                    'follow_up_response': follow_up_response
                })
            
            print("\n" + "="*50)
        
        # Print interview summary
        print("\n=== Interview Summary ===\n")
        for log in interview_log:
            print(f"Question {log['question_number']} - Topic: {', '.join(log['topic'])}")
            print(f"Q1: {log['main_question']}")
            print(f"A1: {log['main_response']}")
            print(f"Q2: {log['follow_up']}")
            print(f"A2: {log['follow_up_response']}\n")

def main():
    job_and_company_info = """
    [input sample job post info and company profile]
    """
    resume = """
    [input sample resume here]
    """

    interviewer = AIInterviewGenerator()
    interviewer.conduct_interview(job_and_company_info, resume, num_questions=3)

if __name__ == "__main__":
    main()