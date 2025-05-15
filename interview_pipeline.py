from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from datetime import datetime
import os
import json

# Configuration - REPLACE WITH YOUR ACTUAL VALUES
QDRANT_DIMENSION = 768
GEMINI_API_KEY = "AIzaSyDE0uhp9RBVE6iPMtL1Va9vziwC-aCs4J0"  # Replace with your actual key
MEM0_API_KEY = "m0-C7KqoRJ7IK8Ogx0RNjhotDkyT0u16yEeX42gja0J"  # Replace with your actual key

# ----------------------
# 1. STATE DEFINITION
# ----------------------
class InterviewState(TypedDict, total=False):
    candidate_id: str
    role: str
    candidate_name: str
    question: str
    answer: str
    feedback: str
    conversation_history: List[BaseMessage]
    resume_context: str
    skill_graph: Dict[str, Any]  # Neo4j skill graph data
    realtime_context: List[str]  # Mem0 retrieved context
    discussed_topics: List[str]  # Track discussed topics

def initialize_neo4j_schema(graph):
    """Initialize Neo4j schema with proper error handling"""
    try:
        # Check and create constraints
        constraints = graph.query("SHOW CONSTRAINTS")
        existing_constraints = {c["name"] for c in constraints}
        
        if "unique_candidate" not in existing_constraints:
            graph.query("""
            CREATE CONSTRAINT unique_candidate IF NOT EXISTS 
            FOR (c:Candidate) REQUIRE c.id IS UNIQUE
            """)
        
        if "unique_skill" not in existing_constraints:
            graph.query("""
            CREATE CONSTRAINT unique_skill IF NOT EXISTS 
            FOR (s:Skill) REQUIRE s.name IS UNIQUE
            """)
            
        # Initialize relationship by creating and deleting a sample
        graph.query("""
        MERGE (dummy:Candidate {id: 'dummy_init'})
        MERGE (dummy_skill:Skill {name: 'dummy_skill'})
        MERGE (dummy)-[r:HAS_SKILL]->(dummy_skill)
        DELETE r, dummy, dummy_skill
        """)
        
        print("✅ Neo4j schema and relationships verified")
    except Exception as e:
        if "EquivalentSchemaRuleAlreadyExists" in str(e):
            print("✅ Neo4j schema already exists")
        else:
            print(f"⚠️ Schema initialization note: {e}")

# ----------------------
# 2. SERVICE INITIALIZATION
# ----------------------
class MemoryManager:
    """Simplified Mem0.ai client for demonstration"""
    def __init__(self):
        self.memory_store = []
    
    def store_context(self, candidate_id: str, context: str, metadata: dict):
        self.memory_store.append({
            "text": context,
            "metadata": {
                "candidate_id": candidate_id,
                "timestamp": datetime.now().isoformat(),
                **metadata
            }
        })
        return True
    
    def retrieve_context(self, candidate_id: str, query: str = None, k: int = 3):
        return sorted(
            [item for item in self.memory_store if item["metadata"]["candidate_id"] == candidate_id],
            key=lambda x: x["metadata"]["timestamp"],
            reverse=True
        )[:k]

def initialize_services():
    """Initialize all external services with error handling"""
    services = {}
    
    try:
        # Initialize LLM
        services['llm'] = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=GEMINI_API_KEY
        )
        
        # Initialize Qdrant
        services['qdrant'] = Qdrant.from_texts(
            texts=[""],
            embedding=GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY
            ),
            url="http://localhost:6333",
            collection_name="resumes",
        )
        
        # Initialize Neo4j
        services['neo4j'] = Neo4jGraph(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )

         # Verify connection with a simple query
        services['neo4j'].query("RETURN 1")
        
         # Initialize schema
        initialize_neo4j_schema(services['neo4j']) 

        # Initialize Memory (using simplified version)
        services['memory'] = MemoryManager()
        
        
    except Exception as e:
        print(f"Service initialization failed: {e}")
        raise
    
    return services

# ----------------------
# 3. PIPELINE NODES
# ----------------------
def load_context(state: InterviewState, services: dict) -> InterviewState:
    """Load initial candidate context"""
    try:
        candidate_id = state["candidate_id"]
        
        # Ensure candidate exists in Neo4j
        services['neo4j'].query("""
        MERGE (c:Candidate {id: $id})
        ON CREATE SET c.created_at = datetime()
        """, {"id": candidate_id})
        
        # Get candidate name
        result = services['neo4j'].query(
            "MATCH (c:Candidate {id: $id}) RETURN c.name AS name",
            {"id": candidate_id}
        )
        candidate_name = result[0]["name"] if result and result[0]["name"] else "Candidate"
        
        # Fetch resume chunks
        docs = services['qdrant'].similarity_search(
            query=f"Resume for {candidate_name} applying for {state['role']}",
            k=3,
            filter=qdrant_models.Filter(
                must=[qdrant_models.FieldCondition(
                    key="metadata.candidate_id",
                    match=qdrant_models.MatchValue(value=candidate_id)
                )]
            )
        )
        
        return {
            **state,
            "candidate_name": candidate_name,
            "resume_context": "\n".join(doc.page_content for doc in docs) if docs else "",
            "conversation_history": [],
            "skill_graph": [],
            "realtime_context": []
        }
    except Exception as e:
        print(f"⚠️ Context loading error: {e}")
        return state
    
def enrich_with_memory(state: InterviewState, services: dict) -> InterviewState:
    """Enrich with real-time context from memory"""
    try:
        # Initialize with empty lists
        memory_results = []
        skill_graph = []
        
        # Only proceed if we have a candidate_id
        if state.get("candidate_id"):
            # Get recent conversation context
            recent_context = "\n".join(
                f"{msg.type}: {msg.content}" 
                for msg in state.get("conversation_history", [])[-3:]
            ) if state.get("conversation_history") else ""
            
            # Retrieve from memory
            memory_results = services['memory'].retrieve_context(
                state["candidate_id"],
                query=recent_context
            ) or []
            
            # Retrieve from Neo4j
            try:
                neo4j_result = services['neo4j'].query("""
                MATCH (c:Candidate {id: $id})-[:HAS_SKILL]->(s:Skill)
                RETURN s.name as skill, count(*) as strength
                ORDER BY strength DESC LIMIT 5
                """, {"id": state["candidate_id"]})
                skill_graph = neo4j_result if neo4j_result else []
            except Exception as e:
                print(f"⚠️ Skill graph query warning: {str(e)[:100]}...")
                skill_graph = []
        
        return {
            **state,
            "realtime_context": [res["text"] for res in memory_results],
            "skill_graph": skill_graph
        }
    except Exception as e:
        print(f"Memory enrichment error: {e}")
        return state


def generate_question(state: InterviewState, services: dict) -> InterviewState:
    """Generate contextual question"""
    try:
        # Safely get context values
        role = state.get("role", "technical role")
        candidate_name = state.get("candidate_name", "Candidate")
        resume_context = state.get("resume_context", "")[:500]
        recent_discussion = state.get("realtime_context", [""])[0][:200]
        skills = [s.get("skill", "") for s in state.get("skill_graph", []) if s]
        
        context = f"""
        Interview for: {role}
        Candidate: {candidate_name}
        Resume Highlights: {resume_context}
        Recent Discussion: {recent_discussion}
        Known Skills: {', '.join(skills) if skills else 'None yet'}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a technical interviewer. Generate one relevant question based on:"),
            ("human", context),
            ("human", "Guidelines:\n- Focus on {role} requirements\n- Relate to mentioned skills if any\n- Ask one clear technical question")
        ])
        
        response = services['llm'].invoke(prompt.format(role=role))
        return {**state, "question": response.content}
        
    except Exception as e:
        print(f"Question generation error: {e}")
        return {**state, "question": "Can you describe your experience with Python?"}

def evaluate_answer(state: InterviewState, services: dict) -> InterviewState:
    """Process candidate answer"""
    try:
        if not state.get("answer"):
            return state
            
        # Update conversation history
        history = state.get("conversation_history", [])
        history.extend([
            HumanMessage(content=state["question"]),
            AIMessage(content=state["answer"])
        ])
        
        return {**state, "conversation_history": history}
    except Exception as e:
        print(f"Evaluation error: {e}")
        return state

def update_knowledge_graph(state: InterviewState, services: dict) -> InterviewState:
    """Update knowledge systems with new information"""
    try:
        if not state.get("answer"):
            return state
            
        # Store in memory
        services['memory'].store_context(
            state["candidate_id"],
            context=f"Q: {state['question']}\nA: {state['answer']}",
            metadata={"role": state["role"], "type": "qa_exchange"}
        )
        
        # Extract skills (simplified)
        skills_prompt = f"Extract technical skills from: {state['answer']}\nReturn as JSON list."
        skills = json.loads(services['llm'].invoke(skills_prompt).content)
        
        # Update Neo4j
        for skill in skills:
            services['neo4j'].query("""
            MERGE (s:Skill {name: $skill})
            MERGE (c:Candidate {id: $id})
            MERGE (c)-[r:HAS_SKILL]->(s)
            ON CREATE SET r.confidence = 1.0
            ON MATCH SET r.confidence = COALESCE(r.confidence, 0) + 0.1
            """, {"skill": skill, "id": state["candidate_id"]})
            
        return state
    except Exception as e:
        print(f"Knowledge update error: {e}")
        return state

def should_evaluate(state: InterviewState) -> bool:
    """Check if evaluation should occur"""
    return "answer" in state and bool(state["answer"])

# ----------------------
# 4. PIPELINE CONSTRUCTION
# ----------------------
def build_interview_pipeline(services: dict):
    """Build and return the interview workflow"""
    workflow = StateGraph(InterviewState)
    
    # Add nodes
    workflow.add_node("initialize", lambda state: load_context(state, services))
    workflow.add_node("enrich", lambda state: enrich_with_memory(state, services))
    workflow.add_node("generate_question", lambda state: generate_question(state, services))
    workflow.add_node("evaluate", lambda state: evaluate_answer(state, services))
    workflow.add_node("update_knowledge", lambda state: update_knowledge_graph(state, services))
    
    # Define workflow
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "enrich")
    workflow.add_edge("enrich", "generate_question")
    
    workflow.add_conditional_edges(
        "generate_question",
        should_evaluate,
        {
            True: "evaluate",
            False: END
        }
    )
    
    workflow.add_edge("evaluate", "update_knowledge")
    
    workflow.add_conditional_edges(
        "update_knowledge",
        lambda state: len(state.get("conversation_history", [])) < 8,
        {
            True: "enrich",  # Get fresh context
            False: END
        }
    )
    
    return workflow.compile()

# ----------------------
# 5. MAIN EXECUTION
# ----------------------
if __name__ == "__main__":
    try:
        # Initialize services with verification
        print("🔄 Initializing services...")
        services = initialize_services()
        
        # Schema verification
        print("\n🔍 Database Schema Status:")
        constraints = services['neo4j'].query("SHOW CONSTRAINTS")
        indexes = services['neo4j'].query("SHOW INDEXES")
        print(f"• Found {len(constraints)} constraints")
        print(f"• Found {len(indexes)} indexes")
        
        # Build pipeline
        print("\n⚙️ Building interview pipeline...")
        pipeline = build_interview_pipeline(services)
        
        # Interview configuration
        initial_state = {
            "candidate_id": "123",
            "role": "Python Developer",
            "answer": "",
            "conversation_history": []
        }
        
        # Run interview
        print("\n" + "="*50)
        print("🎤 Starting AI Interview Session")
        print("Type 'exit' to end early\n" + "="*50)
        
        for step in pipeline.stream(initial_state):
            if "question" in step:
                print(f"\n[INTERVIEWER]: {step['question']}")
                answer = input("[YOU]: ").strip()
                
                if answer.lower() == 'exit':
                    print("\n⚠️ Session ended by user")
                    break
                    
                step["answer"] = answer
            
            # Display progress
            history = step.get("conversation_history", [])
            print(f"\n📊 Progress: {len(history)//2} questions completed")
            
        # Post-interview summary
        print("\n" + "="*50)
        print("📝 Interview Summary")
        print(f"• Total questions: {len(initial_state['conversation_history'])//2}")
        
        # Save to Neo4j
        try:
            services['neo4j'].query("""
            MERGE (i:Interview {id: $interview_id})
            SET i.role = $role, i.date = datetime()
            """, {
                "interview_id": f"int_{initial_state['candidate_id']}_{datetime.now().timestamp()}",
                "role": initial_state["role"]
            })
            print("• Interview saved to Neo4j")
        except Exception as e:
            print(f"• Could not save to Neo4j: {str(e)}")
            
        print("="*50 + "\n")
        print("✅ Interview completed successfully!")

    except KeyboardInterrupt:
        print("\n🛑 Interview session interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        print("Please check your service connections and try again")