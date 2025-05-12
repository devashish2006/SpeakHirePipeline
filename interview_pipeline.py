from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Qdrant  # Updated import
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os

# Configuration
# GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("GOOGLE_API_KEY environment variable not set")

QDRANT_DIMENSION = 768

# Define state schema
class InterviewState(TypedDict, total=False):
    candidate_id: str
    role: str
    candidate_name: str
    question: str
    answer: str
    feedback: str
    conversation_history: List[BaseMessage]
    resume_context: str

# Initialize components
try:
    # Initialize LLM - Modified to use direct API key
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key="AIzaSyDE0uhp9RBVE6iPMtL1Va9vziwC-aCs4J0"
    )
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyDE0uhp9RBVE6iPMtL1Va9vziwC-aCs4J0"
    )
    
    # Initialize Qdrant
    qdrant = Qdrant.from_texts(
    texts=[""],  # Dummy text (can be empty if you're loading an existing collection)
    embedding=embeddings,  # Pass the embedding model here
    url="http://localhost:6333",  # Qdrant server URL
    collection_name="resumes",
)
    
    # Initialize Neo4j
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    
except Exception as e:
    print(f"Initialization error: {e}")
    raise

def load_context(state: InterviewState) -> InterviewState:
    try:
        candidate_id = state["candidate_id"]
        result = graph.query(f"""
            MATCH (c:Candidate {{id: '{candidate_id}'}})
            RETURN c.name AS name
        """)
        candidate_name = result[0]["name"] if result else "Candidate"
        
        # Fetch resume chunks from Qdrant (filter by candidate_id)
        docs = qdrant.similarity_search(
            query=f"Resume of {candidate_name} applying for {state['role']}",
            k=3,
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.candidate_id",
                        match=qdrant_models.MatchValue(value=candidate_id),
                    )  # <-- Fixed missing parenthesis
                ]
            )
        )
        resume_context = "\n".join([doc.page_content for doc in docs])
        
        return {
            **state,
            "candidate_name": candidate_name,
            "resume_context": resume_context,
            "conversation_history": []
        }
    except Exception as e:
        print(f"⚠️ Load context error: {e}")
        return state

def generate_question(state: InterviewState) -> InterviewState:
    try:
        role = state["role"]
        resume_context = state.get("resume_context", "")
        history = state.get("conversation_history", [])
        
        # Format conversation history
        history_str = "\n".join(
            f"{'Interviewer' if isinstance(msg, AIMessage) else 'Candidate'}: {msg.content}"
            for msg in history[-4:]
        )
        
        prompt = f"""You are a technical interviewer conducting an interview for a {role} position.
        
        Resume Context:
        {resume_context}
        
        Conversation History:
        {history_str}
        
        Ask a technical question about a DIFFERENT project or skill than previously discussed or related to the role.
        Focus on concrete implementation details and challenges.
        Ask only one clear question without feedback or suggestions.
        Maintain professional tone.
        font just ask questions, bdo some very sligh comedy, or check the confidence if the candidate 
        is not sure about the answer, and ask him to be sure about the answer.
        
        Example formats:
        you can also ask for one oneWord questions , and wil judge as are you sure ..
        - "Your resume mentions [TECHNOLOGY] in [PROJECT]. How did you handle [SPECIFIC CHALLENGE]?"
        - "For [PROJECT], what was your approach to [TECHNICAL ASPECT]?"
        - "Tell me about a time you used [SKILL] to solve [PROBLEM TYPE] in [PROJECT]"
        - "you said u will use [SKILL] but what if [PROBLEM] happens?"
        - "have u done this [TECHNOLOGY] before? if yes, how did u do it?"
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = gemini_llm.invoke(messages)
        question = response.content
        
        # Ensure we don't repeat projects
        if any(project in question for project in state.get("discussed_projects", [])):
            return generate_question(state)  # Recursively get new question
        
        return {
            **state, 
            "question": question,
            "discussed_projects": state.get("discussed_projects", []) + [
                proj for proj in ["TrueFeedback", "OtherProject"] if proj in question
            ]
        }
        
    except Exception as e:
        print(f"Question generation error: {e}")
        return {**state, "question": "Can you describe a challenging technical problem you've solved?"}

def evaluate_answer(state: InterviewState) -> InterviewState:
    try:
        answer = state.get("answer", "")
        question = state.get("question", "")
        resume_context = state.get("resume_context", "")
        
        if not answer:
            return {**state, "feedback": ""}  # No feedback during interview
        
        # Generate follow-up question based on answer
        prompt = f"""Based on this interview exchange:
        
        Question: {question}
        Answer: {answer}
        
        Resume Context:
        {resume_context}
        
        Ask ONE technical follow-up question that:
        1. Probes deeper into a different aspect of the same technology
        2. OR transitions to a related but different technology from their resume
        3. Avoids giving feedback or suggestions
        4. Maintains natural interview flow
        """
        
        messages = [{"role": "user", "content": prompt}]
        follow_up = gemini_llm.invoke(messages).content
        
        # Update history without feedback
        history = state.get("conversation_history", [])
        history.extend([
            HumanMessage(content=state["question"]),
            AIMessage(content=answer)
        ])
        
        return {
            **state,
            "question": follow_up,
            "conversation_history": history
        }
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {**state, "question": "Let's move to another topic..."}

def should_evaluate(state: InterviewState) -> bool:
    return "answer" in state and bool(state["answer"])

def build_interview_pipeline():
    workflow = StateGraph(InterviewState)
    
    workflow.add_node("LoadContext", load_context)
    workflow.add_node("GenerateQuestion", generate_question)
    workflow.add_node("EvaluateAnswer", evaluate_answer)
    
    workflow.set_entry_point("LoadContext")
    workflow.add_edge("LoadContext", "GenerateQuestion")
    
    workflow.add_conditional_edges(
        "GenerateQuestion",
        should_evaluate,
        {
            True: "EvaluateAnswer",
            False: END
        }
    )
    
    workflow.add_conditional_edges(
        "EvaluateAnswer",
        lambda state: len(state.get("conversation_history", [])) < 10,
        {
            True: "GenerateQuestion",
            False: END
        }
    )
    
    return workflow.compile()