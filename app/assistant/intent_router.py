from app.llm.summarizer import GeminiClient


class IntentRouter:
    """
    Uses Gemini to classify user intent into
    a controlled set of allowed assistant actions.
    """

    def __init__(self):
        self.llm = GeminiClient()

        # Strict allowed actions
        self.allowed_actions = [
            "continue_briefing",
            "deep_dive_current",
            "save_current",
            "casual_conversation",
            "stop",
        ]

    def route(self, user_input: str, state_info: str) -> str:
        """
        Routes user input into one allowed action.
        If uncertain, defaults to 'casual_conversation'.
        """

        prompt = f"""
You are an intent classification system for an AI news assistant.

Conversation state:
{state_info}

User input:
"{user_input}"

Available actions:
{self.allowed_actions}

Action definitions:
- continue_briefing → move to the next AI story
- deep_dive_current → provide technical details about current story
- save_current → mark current story as important
- casual_conversation → respond conversationally about current story
- stop → end the briefing session

Rules:
- Return ONLY one action name from the available list.
- Do NOT explain your reasoning.
- Do NOT output anything except the action word.
- If unsure, return "casual_conversation".
"""

        try:
            response = self.llm.generate(prompt).strip().lower()
        except Exception:
            # Fallback if API fails
            return "casual_conversation"

        # Safety check: enforce strict action list
        if response in self.allowed_actions:
            return response

        return "casual_conversation"