from interview_pipeline import build_interview_pipeline
from langchain_core.messages import HumanMessage, AIMessage

def main():
    pipeline = build_interview_pipeline()
    
    print("Technical Interview Simulation")
    print("=" * 40)
    print("Please answer questions about your resume experience.")
    print("Type 'exit' to end the interview.\n")
    
    state = {
        "candidate_id": "candidate_1",
        "role": "Software Engineer",
        "conversation_history": [],
        "discussed_projects": []
    }
    
    # Start interview
    current_state = pipeline.invoke(state)
    print(f"Interviewer: {current_state.get('question')}")
    
    while True:
        answer = input("\nYour answer: ").strip()
        if answer.lower() == 'exit':
            break
            
        current_state = pipeline.invoke({**current_state, "answer": answer})
        print(f"\nInterviewer: {current_state.get('question')}")
    
    print("\nInterview concluded. Thank you!")

if __name__ == "__main__":
    main()