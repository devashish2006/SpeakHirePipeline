from resume_parser import process_resume
from audio_transcribe import record_audio, transcribe_audio
from interview_pipeline import build_interview_pipeline
import uuid

# Step 1: Parse resume and load data
candidate_id = str(uuid.uuid4())  # Generates a unique UUID
process_resume("resume_samples/candidate1.pdf", candidate_id)

# Step 2: Start interview
pipeline = build_interview_pipeline()

# Role can be hardcoded or from input
state = {"candidate_id": candidate_id, "role": "Software Engineer"}
state = pipeline.invoke(state)

# Ask question
print("\n🧠 Interview Question:\n", state["question"])

# Step 3: Record and transcribe candidate answer
record_audio()
answer_text = transcribe_audio()
print("\n🎤 Candidate Answer:\n", answer_text)

# Step 4: Evaluate answer
state["answer"] = answer_text
state = pipeline.invoke(state)

print("\n✅ Feedback:\n", state["feedback"])
