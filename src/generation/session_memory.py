import uuid
from typing import List, Dict, Optional

class SessionMemory:
    """In-memory session storage for chat history."""
    
    def __init__(self):
        # Format: { session_id: [{"role": "user"/"assistant", "content": "..."}] }
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def create_session(self) -> str:
        """Create a new session and return the UUID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the session's history."""
        if session_id not in self.sessions:
            # According to requirement: handle non-existent session explicitly
            raise ValueError(f"Session '{session_id}' does not exist.")
            
        if role not in ("user", "assistant", "system"):
            raise ValueError("Role must be one of: 'user', 'assistant', 'system'.")
            
        self.sessions[session_id].append({
            "role": role,
            "content": content
        })

    def get_history(self, session_id: str, max_turns: int = 5) -> List[Dict[str, str]]:
        """
        Get chat history for a session, limited to the last `max_turns` interactions.
        1 turn = 1 user message + 1 assistant message (usually).
        We'll just slice the last `max_turns * 2` messages to be safe.
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session '{session_id}' does not exist.")
            
        history = self.sessions[session_id]
        
        # We limit the raw number of messages to max_turns * 2 
        # (assuming typical alternating user/assistant messages)
        max_messages = max_turns * 2
        
        return history[-max_messages:] if len(history) > max_messages else history.copy()
