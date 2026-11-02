from app.retrieval.retrieval_engine import RetrievalEngine
from app.llm.summarizer import GeminiClient
from app.assistant.intent_router import IntentRouter
from app.memory.story_state_store import StoryStateStore
import re
from datetime import datetime, timedelta
from app.voice.voice_handler import VoiceHandler

    
class CLIAssistant:
    def __init__(self, store):
        self.store = store
        self.engine = RetrievalEngine(self.store)
        self.llm = GeminiClient()
        self.router = IntentRouter()
        self.state = StoryStateStore()
        self.state.init()
        self.voice_handler = VoiceHandler()

        self.stories = []
        self.current_index = 0
        self.current_story = None
        self.session_active = False
        self._wake_requested_news = False

    @staticmethod
    def _has_usable_story_text(story: dict) -> bool:
        text = ((story or {}).get("document") or "").strip()
        return len(text) >= 40

    def _sanitize_story_list(self, stories):
        return [story for story in (stories or []) if self._has_usable_story_text(story)]

    # ---------------------------------------
    # ENTRY
    # ---------------------------------------

    def start(self):
        print("\nAI Assistant:")
        self.wake_up_job()

    def _classify_yes_no(self, user_answer: str) -> str:
        """Use the LLM to classify natural spoken replies as yes/no."""
        normalized = (user_answer or "").strip()
        if not normalized:
            return "UNCLEAR"

        prompt = f"""
Classify this user reply to a yes/no question.

Reply with exactly one word:
- YES
- NO
- UNCLEAR

User reply: "{normalized}"
"""
        try:
            label = (self.llm.generate(prompt) or "").strip().upper()
            if "YES" in label:
                return "YES"
            if "NO" in label:
                return "NO"
        except Exception:
            pass

        lowered = normalized.lower()
        yes_words = ("yes", "yeah", "yep", "yup", "sure", "of course", "please do")
        no_words = ("no", "nope", "nah", "not now", "later", "stop")
        if any(word in lowered for word in yes_words):
            return "YES"
        if any(word in lowered for word in no_words):
            return "NO"
        return "UNCLEAR"

    def _classify_confirmation(self, user_reply: str) -> str:
        """Classify whether user confirms or rejects the assistant's interpretation."""
        normalized = (user_reply or "").strip()
        if not normalized:
            return "UNCLEAR"

        prompt = f"""
Classify this user reply after the assistant asks:
"Did I hear you correctly?"

Reply with exactly one word:
- CONFIRM
- REJECT
- UNCLEAR

User reply: "{normalized}"
"""
        try:
            label = (self.llm.generate(prompt) or "").strip().upper()
            if "CONFIRM" in label:
                return "CONFIRM"
            if "REJECT" in label:
                return "REJECT"
        except Exception:
            pass

        lowered = normalized.lower()
        confirm_words = (
            "correct",
            "that's correct",
            "yes correct",
            "right",
            "exactly",
            "yes",
        )
        reject_words = (
            "wrong",
            "that's wrong",
            "no",
            "not correct",
            "incorrect",
        )
        if any(word in lowered for word in confirm_words):
            return "CONFIRM"
        if any(word in lowered for word in reject_words):
            return "REJECT"
        return "UNCLEAR"

    def _ask_yes_no(self, question: str) -> bool:
        """
        Ask a yes/no question, confirm interpretation with user,
        then decide only after confirmation.
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            if attempt == 0:
                answer = self.voice_handler.listen(question).strip()
            else:
                answer = self.voice_handler.listen(
                    "Let's try again. Please answer yes or no."
                ).strip()

            decision = self._classify_yes_no(answer)
            if decision not in {"YES", "NO"}:
                self.voice_handler.speak(
                    "I could not clearly understand that. Let's try once more."
                )
                continue

            heard_value = "yes" if decision == "YES" else "no"
            confirmation = self.voice_handler.listen(
                f"I heard {heard_value}. Is that correct? "
                "Please say that's correct, or that's wrong."
            ).strip()
            confirmation_decision = self._classify_confirmation(confirmation)

            if confirmation_decision == "CONFIRM":
                print(f"Heard: {heard_value}. Confirmed.")
                self.voice_handler.speak(f"Thanks. I confirmed {heard_value}.")
                return decision == "YES"

            if confirmation_decision == "REJECT":
                self.voice_handler.speak("Thanks for correcting me. Let's try again.")
                continue

            self.voice_handler.speak(
                "I did not get a clear confirmation. Let's try again."
            )

        print("Could not get a confirmed yes or no.")
        self.voice_handler.speak("I could not confirm your answer. We can try again.")
        return False

    def _classify_wake_intent(self, user_reply: str) -> str:
        """
        Classify first wake response into:
        AWAKE_AND_NEWS, AWAKE_CHAT, NOT_READY, UNCLEAR.
        """
        normalized = (user_reply or "").strip()
        if not normalized:
            return "UNCLEAR"

        prompt = f"""
Classify this wake-up reply.

Return exactly one label:
- AWAKE_AND_NEWS: user is awake and wants AI news now.
- AWAKE_CHAT: user is awake but chatting or not requesting news yet.
- NOT_READY: user wants to sleep, stop, or do this later.
- UNCLEAR: cannot decide.

User reply: "{normalized}"
"""
        try:
            label = (self.llm.generate(prompt) or "").strip().upper()
            for allowed in ("AWAKE_AND_NEWS", "AWAKE_CHAT", "NOT_READY", "UNCLEAR"):
                if allowed in label:
                    return allowed
        except Exception:
            pass

        lowered = normalized.lower()
        if any(word in lowered for word in ("later", "not now", "sleep", "stop")):
            return "NOT_READY"
        if any(
            phrase in lowered
            for phrase in ("ai news", "news", "updates", "briefing", "update me")
        ):
            return "AWAKE_AND_NEWS"
        if any(word in lowered for word in ("yes", "awake", "morning", "hi", "hello")):
            return "AWAKE_CHAT"
        return "UNCLEAR"

    def _classify_wake_chat_intent(self, user_reply: str) -> str:
        """
        Classify wake follow-up into:
        REQUEST_NEWS, NOT_NOW, CHAT, UNCLEAR.
        """
        normalized = (user_reply or "").strip()
        if not normalized:
            return "UNCLEAR"

        prompt = f"""
Classify this user reply in wake-up conversation.

Return exactly one label:
- REQUEST_NEWS: user wants AI updates now.
- NOT_NOW: user wants to stop, sleep, or do news later.
- CHAT: user is casually chatting.
- UNCLEAR: cannot decide.

User reply: "{normalized}"
"""
        try:
            label = (self.llm.generate(prompt) or "").strip().upper()
            for allowed in ("REQUEST_NEWS", "NOT_NOW", "CHAT", "UNCLEAR"):
                if allowed in label:
                    return allowed
        except Exception:
            pass

        lowered = normalized.lower()
        if any(
            phrase in lowered
            for phrase in ("ai news", "news", "updates", "briefing", "start news")
        ):
            return "REQUEST_NEWS"
        if any(word in lowered for word in ("later", "not now", "sleep", "stop")):
            return "NOT_NOW"
        return "CHAT"

    def _confirm_interpretation(self, interpreted_text: str) -> bool:
        """Ask user to confirm what assistant understood."""
        reply = self.voice_handler.listen(
            f"I heard: {interpreted_text}. Is that correct?"
        ).strip()
        if not reply:
            return False

        decision = self._classify_confirmation(reply)
        if decision == "CONFIRM":
            return True
        if decision == "REJECT":
            return False
        return self._classify_yes_no(reply) == "YES"

    def _wake_casual_reply(self, user_input: str) -> None:
        """Short pre-news conversation not tied to story context."""
        prompt = f"""
You are a warm morning voice assistant.
User is waking up. Keep your reply short and natural (1-2 sentences).
Be friendly and gently suggest AI updates when appropriate.

User says:
{user_input}
"""
        response = self.llm.generate(prompt)
        print("\nAssistant:\n")
        print(response)
        self.voice_handler.speak(response)

    def _run_wake_conversation(self, seed_input: str = "") -> str:
        """
        Run a short wake conversation.
        Returns REQUEST_NEWS, NOT_NOW, or CONTINUE_TO_NEWS_QUESTION.
        """
        if seed_input:
            self._wake_casual_reply(seed_input)

        max_turns = 3
        for _ in range(max_turns):
            user_input = self.voice_handler.listen(
                "We can chat for a bit. When you are ready, say give me AI news."
            ).strip()

            if not user_input:
                self.voice_handler.speak(
                    "I did not hear a response. I will check again later."
                )
                return "NOT_NOW"

            intent = self._classify_wake_chat_intent(user_input)
            if intent == "REQUEST_NEWS":
                return "REQUEST_NEWS"
            if intent == "NOT_NOW":
                return "NOT_NOW"

            self._wake_casual_reply(user_input)

        return "CONTINUE_TO_NEWS_QUESTION"

    def ask_awake(self) -> bool:
        self._wake_requested_news = False
        first_reply = self.voice_handler.listen(
            "Good morning, Abhi. Are you awake?", timeout=15
        ).strip()
        if not first_reply:
            self.voice_handler.speak(
                "I did not hear a response. I will check again later."
            )
            return False

        wake_intent = self._classify_wake_intent(first_reply)

        if wake_intent == "NOT_READY":
            return False

        if wake_intent == "AWAKE_AND_NEWS":
            if self._confirm_interpretation("you are awake and want AI news now"):
                self._wake_requested_news = True
                return True
            wake_result = self._run_wake_conversation(seed_input=first_reply)
            if wake_result == "REQUEST_NEWS":
                self._wake_requested_news = True
                return True
            return wake_result == "CONTINUE_TO_NEWS_QUESTION"

        if wake_intent == "AWAKE_CHAT":
            if not self._confirm_interpretation("you are awake"):
                clarified = self.voice_handler.listen(
                    "Please say yes if you are awake."
                ).strip()
                if self._classify_yes_no(clarified) != "YES":
                    return False

            wake_result = self._run_wake_conversation(seed_input=first_reply)
            if wake_result == "REQUEST_NEWS":
                self._wake_requested_news = True
                return True
            return wake_result == "CONTINUE_TO_NEWS_QUESTION"

        clarified = self.voice_handler.listen(
            "I could not tell clearly. Are you awake? Please answer yes or no."
        ).strip()
        if self._classify_yes_no(clarified) != "YES":
            return False
        return True

    def ask_latest_updates(self) -> bool:
        answer = self.voice_handler.listen(
            "Should I give you the latest AI updates?"
        ).strip()
        if not answer:
            self.voice_handler.speak(
                "I did not hear a response. I will check again later."
            )
            return False

        intent = self._classify_wake_chat_intent(answer)
        if intent == "REQUEST_NEWS":
            return self._confirm_interpretation("you want AI updates now")
        if intent == "NOT_NOW":
            return False

        return self._ask_yes_no(
            "I am not fully sure. Do you want AI updates now? Please say yes or no."
        )

    def wake_up_job(self) -> None:
        notify = self.state.get_important_notify_status(limit=20)
        if notify.should_notify:
            print(
                f"Note: important stories count is {notify.current_count} "
                f"(limit {notify.limit})."
            )
            self.state.ack_important_notification(notify.current_count, step=15)
        if not self.ask_awake():
            print("No problem. I'll check again later.")
            return

        if self._wake_requested_news:
            self.handle_news_request()
            return

        if not self.ask_latest_updates():
            print("Okay. I'll wait until later.")
            return

        self.handle_news_request()
    # def start(self):
    #     print("\nAI Assistant:")

    #     notify = self.state.get_important_notify_status(limit=20)
    #     if notify.should_notify:
    #         print(
    #             f"Note: important stories count is {notify.current_count} "
    #             f"(limit {notify.limit})."
    #         )
    #         print("Important deletion UI is not implemented yet.")
    #         self.state.ack_important_notification(notify.current_count, step=15)

    #     awake = input("Are you awake? (yes/no): ").strip().lower()

    #     if awake != "yes":
    #         print("Alright. I'll check again later.")
    #         return

    #     self.handle_news_request()

    # ---------------------------------------
    # FETCH STORIES
    # ---------------------------------------

    def handle_news_request(self):
        user_query = self.voice_handler.listen(
            "Would you like today's AI update or something specific?"
        ).strip()

        normalized = (user_query or "").strip().lower()

        day_key = None
        if "yesterday" in normalized:
            yesterday = datetime.now().astimezone() - timedelta(days=1)
            day_key = yesterday.strftime("%Y-%m-%d")
        else:
            m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", normalized)
            if m:
                day_key = m.group(1)

        is_today_request = (
            not normalized
            or "today" in normalized
            or normalized in {"latest", "update", "ai update"}
            or "ai update" in normalized
            or "ai news" in normalized
            or (
                "update" in normalized
                and len([tok for tok in re.split(r"\s+", normalized) if tok]) <= 5
            )
        )

        if day_key is not None:
            stories = self.engine.retrieve_distinct_stories_for_day(day_key=day_key)
        # Default "today" flow: fetch recent articles.
        # Deduplicate via clustering.
        elif is_today_request:
            stories = self.engine.retrieve_distinct_stories_today()
        else:
            stories = self.engine.retrieve_distinct_stories(query=user_query)

        stories = self._sanitize_story_list(stories)
        if not stories and is_today_request:
            stories = self._sanitize_story_list(
                self.engine.retrieve_distinct_stories_last_hours(hours=48)
            )

        if not stories:
            print("No relevant AI stories found.")
            return

        if day_key is not None:
            print(f"\nThere are {len(stories)} distinct AI stories for {day_key}.")
        elif is_today_request:
            print(f"\nThere are {len(stories)} distinct AI stories today.")
        else:
            print(
                f"\nThere are {len(stories)} distinct AI stories matching your request."
            )
        print("Starting with the first one.")

        self.stories = stories
        self.current_index = 0
        self.session_active = True

        self._present_current_story()
        self._run_session_loop()

    # ---------------------------------------
    # MAIN EVENT LOOP (Voice-Ready)
    # ---------------------------------------

    def _run_session_loop(self):
        while self.session_active:
            user_input = self.voice_handler.listen(
                "What would you like to do next?"
            ).strip()

            if self._handle_direct_command(user_input):
                continue

            state_info = self._build_state_info()
            action = self.router.route(user_input, state_info)

            self._execute_action(action, user_input)

    def _handle_direct_command(self, user_input: str) -> bool:
        normalized = (user_input or "").strip().lower()

        if normalized.startswith("delete important"):
            m = re.search(r"delete important\s+(\d+)", normalized)
            n = int(m.group(1)) if m else 15
            self._delete_oldest_important(n)
            return True

        if (
            "session" in normalized
            and any(word in normalized for word in ("end", "and", "stop", "close", "quit", "finish"))
        ) or normalized in {"stop", "end", "quit"}:
            self._confirm_stop()
            return True

        return False

    def _delete_oldest_important(self, n: int) -> None:
        if n <= 0:
            print("Please provide a positive number, e.g. 'delete important 15'.")
            return

        current = self.state.count_important()
        if current <= 0:
            print("There are no important stories saved.")
            return

        to_delete = min(n, current)
        confirm = (
            self.voice_handler.listen(
                f"This will delete the {to_delete} oldest important stories. Continue? Please say yes or no."
            )
            .strip()
            .lower()
        )

        if confirm != "yes":
            print("OK. Keeping all important stories.")
            return

        ids = self.state.list_oldest_important_ids(limit=to_delete)
        if not ids:
            print("No important stories found to delete.")
            return

        member_map = self.store.get_distinct_member_article_ids_map(ids)
        raw_ids = []
        for member_ids in member_map.values():
            raw_ids.extend(member_ids or [])

        self.store.delete_raw_articles_by_ids(sorted(set(raw_ids)))
        self.store.delete_distinct_stories_by_ids(ids)
        self.state.delete_story_states(ids)

        print(f"Deleted {len(ids)} important stories.")

    # ---------------------------------------
    # STATE INFO
    # ---------------------------------------

    def _build_state_info(self):
        return f"""
Active story:
{self.current_story['headline'] if self.current_story else 'None'}

Stories remaining:
{len(self.stories) - self.current_index - 1}
"""

    # ---------------------------------------
    # STORY PRESENTATION
    # ---------------------------------------

    def _present_current_story(self):
        if self.current_index >= len(self.stories):
            print("\nThat concludes today's AI stories.")
            self.session_active = False
            return

        story = self.stories[self.current_index]
        if not self._has_usable_story_text(story):
            print("Skipping one story because the text content is missing.")
            self.current_index += 1
            self._present_current_story()
            return

        self.current_story = story

        distinct_id = story.get("distinct_story_id")
        day_key = (story.get("metadata") or {}).get("day_key")
        if distinct_id and day_key:
            self.state.ensure_unheard(distinct_id, day_key)
            self.state.mark_presented(distinct_id)

        prompt = f"""
Summarize this AI story conversationally in 2–3 sentences.
Avoid deep technical detail.

Story:
{story['document']}
"""

        summary = self.llm.generate(prompt)

        print(f"\nStory {self.current_index + 1}:")
        print(summary)
        self.voice_handler.speak(summary)

    # ---------------------------------------
    # ACTION EXECUTION
    # ---------------------------------------

    def _execute_action(self, action, user_input):

        if action == "continue_briefing":
            self._mark_current_story_heard()
            print("Alright, moving to the next story.")
            self.current_index += 1
            self._present_current_story()

        elif action == "deep_dive_current":
            print("Alright, let's go deeper.")
            self._deep_dive()

        elif action == "save_current":
            print("Got it. Saving this story as important.")
            self._save_current_article()

        elif action == "casual_conversation":
            self._casual_reply(user_input)

        elif action == "stop":
            self._confirm_stop()

        else:
            self._casual_reply(user_input)

    # ---------------------------------------
    # DEEP DIVE
    # ---------------------------------------

    def _deep_dive(self):
        prompt = f"""
Provide a technical explanation of this AI story.
Stay grounded in the provided content.

Story:
{self.current_story['document']}
"""

        response = self.llm.generate(prompt)

        print("\nTechnical Deep Dive:\n")
        print(response)
        self.voice_handler.speak(response)

    # ---------------------------------------
    # CASUAL CONVERSATION
    # ---------------------------------------

    def _casual_reply(self, user_input):
        prompt = f"""
We are discussing this AI story:

{self.current_story['document']}

User says:
{user_input}

Respond conversationally.
You may expand slightly beyond the article if helpful.
"""

        response = self.llm.generate(prompt)

        print("\nAssistant:\n")
        print(response)
        self.voice_handler.speak(response)

    # ---------------------------------------
    # SAVE
    # ---------------------------------------

    def _save_current_article(self):
        distinct_id = (self.current_story or {}).get("distinct_story_id")
        day_key = ((self.current_story or {}).get("metadata") or {}).get("day_key")
        if not distinct_id or not day_key:
            print("This story cannot be saved (missing distinct story ID).")
            return

        self.state.ensure_unheard(distinct_id, day_key)
        self.state.mark_important(distinct_id)

        notify = self.state.get_important_notify_status(limit=20)
        if notify.should_notify:
            print(
                f"Note: important stories count is {notify.current_count} "
                f"(limit {notify.limit})."
            )
            print(
                "You can keep adding; you'll be notified again "
                "after 15 more are added."
            )
            self.state.ack_important_notification(notify.current_count, step=15)

    # ---------------------------------------
    # SAFE STOP CONFIRMATION
    # ---------------------------------------

    def _confirm_stop(self):
        confirm = (
            self.voice_handler.listen(
                "Are you sure you want to end the session? Please say yes or no."
            )
            .strip()
        )

        if self._classify_yes_no(confirm) == "YES":
            self._mark_current_story_heard()
            print("Alright. Ending session.")
            self.session_active = False
        else:
            print("Continuing briefing.")

    def _mark_current_story_heard(self):
        distinct_id = (self.current_story or {}).get("distinct_story_id")
        if distinct_id:
            self.state.mark_heard(distinct_id)

