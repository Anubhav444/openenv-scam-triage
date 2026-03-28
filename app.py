from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import random

app = FastAPI(title="Scam Triage OpenEnv")

# --- 1. Typed Models (OpenEnv Spec) ---
class Observation(BaseModel):
    message_id: str
    sender: str
    content: str
    task_level: str

class Action(BaseModel):
    is_scam: bool
    threat_category: Optional[str] = None
    extracted_url: Optional[str] = None
    quarantine_action: bool

class Reward(BaseModel):
    score: float
    feedback: str
    done: bool

# --- 2. Mock Real-World Data ---
MOCK_DATA = [
    {"id": "msg1", "sender": "support@paypa1.com", "content": "Your account is locked. Verify here: http://bit.ly/secure-login-12", "scam": True, "cat": "phishing", "url": "http://bit.ly/secure-login-12"},
    {"id": "msg2", "sender": "mom@gmail.com", "content": "Can you call me after work? Need help with the router.", "scam": False, "cat": "safe", "url": None},
    {"id": "msg3", "sender": "unknown_number", "content": "Urgent! You won the lottery. Send processing fee to unlock funds.", "scam": True, "cat": "social_engineering", "url": None}
]

# --- 3. Environment State ---
class ScamTriageEnv:
    def __init__(self):
        self.current_message = None
        self.task_level = "easy"
        self.history = []
        self.reset()

    def reset(self, level: str = "easy") -> Observation:
        self.task_level = level
        self.current_message = random.choice(MOCK_DATA)
        self.history = []
        return self.state()

    def state(self) -> Observation:
        if not self.current_message:
            self.reset()
        return Observation(
            message_id=self.current_message["id"],
            sender=self.current_message["sender"],
            content=self.current_message["content"],
            task_level=self.task_level
        )

    def step(self, action: Action) -> Reward:
        truth = self.current_message
        score = 0.0
        feedback = []

        if action.is_scam == truth["scam"]:
            score += 0.4
            feedback.append("Correct binary classification.")
        else:
            feedback.append("Incorrect binary classification.")

        if self.task_level in ["medium", "hard"]:
            if action.threat_category == truth["cat"]:
                score += 0.3
                feedback.append("Correct threat category.")
            else:
                feedback.append(f"Expected category {truth['cat']}.")

        if self.task_level == "hard":
            if action.extracted_url == truth["url"]:
                score += 0.15
                feedback.append("Correct URL extraction.")
            if action.quarantine_action == truth["scam"]:
                score += 0.15
                feedback.append("Correct quarantine decision.")

        if self.task_level == "easy":
            final_score = score / 0.4 if score > 0 else 0.0
        elif self.task_level == "medium":
            final_score = score / 0.7 if score > 0 else 0.0
        else:
            final_score = score

        self.history.append({"action": action.dict(), "score": final_score})
        
        return Reward(
            score=round(final_score, 2),
            feedback=" | ".join(feedback),
            done=True
        )

env = ScamTriageEnv()

# --- 4. OpenEnv Endpoints ---
@app.post("/reset", response_model=Observation)
def reset_env(level: str = "easy"):
    return env.reset(level)

@app.get("/state", response_model=Observation)
def get_state():
    return env.state()

@app.post("/step", response_model=Reward)
def take_step(action: Action):
    return env.step(action)

# --- 5. Hackathon Required Endpoints ---
@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {"id": "task_1", "difficulty": "easy", "description": "Identify if the message is a scam."},
            {"id": "task_2", "difficulty": "medium", "description": "Identify the scam and categorize the threat type."},
            {"id": "task_3", "difficulty": "hard", "description": "Identify, categorize, extract URLs, and quarantine."}
        ],
        "action_schema": Action.schema()
    }

@app.get("/grader")
def get_grader():
    if not env.history:
        return {"error": "No episodes completed yet."}
    return {"latest_score": env.history[-1]["score"]}

@app.post("/baseline")
def run_baseline():
    return {"baseline_scores": {"easy": 1.0, "medium": 0.85, "hard": 0.70}}
