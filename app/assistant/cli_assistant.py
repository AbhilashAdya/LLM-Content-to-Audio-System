from app.retrieval.retrieval_engine import RetrievalEngine
from app.llm.summarizer import GeminiClient
from app.assistant.intent_router import IntentRouter
from app.memory.story_state_store import StoryStateStore
import re
from datetime import datetime, timedelta


class CLIAssistant:
    def __init__(self, store):
        self.store = store
        self.engine = RetrievalEngine(self.store)
        self.llm = GeminiClient()
        self.router = IntentRouter()
        self.state = StoryStateStore()
        self.state.init()

        self.stories = []
        self.current_index = 0
        self.current_story = None
        self.session_active = False

    # ---------------------------------------
    # ENTRY
    # ---------------------------------------

    def start(self):
        print("\nAI Assistant:")

        notify = self.state.get_important_notify_status(limit=20)
        if notify.should_notify:
            print(
                f"Note: important stories count is {notify.current_count} "
                f"(limit {notify.limit})."
            )
            print("Important deletion UI is not implemented yet.")
            self.state.ack_important_notification(notify.current_count, step=15)

        awake = input("Are you awake? (yes/no): ").strip().lower()

        if awake != "yes":
            print("Alright. I'll check again later.")
            return

        self.handle_news_request()

    # ---------------------------------------
    # FETCH STORIES
    # ---------------------------------------

    def handle_news_request(self):
        print("\nWould you like today's AI update or something specific?")
        user_query = input("> ").strip()

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
        )

        if day_key is not None:
            stories = self.engine.retrieve_distinct_stories_for_day(day_key=day_key)
        # Default "today" flow: fetch recent articles.
        # Deduplicate via clustering.
        elif is_today_request:
            stories = self.engine.retrieve_distinct_stories_today()
        else:
            stories = self.engine.retrieve_distinct_stories(query=user_query)

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
            user_input = input("\n> ").strip()

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
        confirm = input(
            f"This will delete the {to_delete} oldest important stories. "
            "Continue? (yes/no): "
        ).strip().lower()

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
        confirm = input(
            "Are you sure you want to end the session? (yes/no): "
        ).strip().lower()

        if confirm == "yes":
            self._mark_current_story_heard()
            print("Alright. Ending session.")
            self.session_active = False
        else:
            print("Continuing briefing.")

    def _mark_current_story_heard(self):
        distinct_id = (self.current_story or {}).get("distinct_story_id")
        if distinct_id:
            self.state.mark_heard(distinct_id)
