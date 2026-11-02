import time
from apscheduler.schedulers.background import BackgroundScheduler

from app.assistant.cli_assistant import CLIAssistant
from app.ingestion.run_ingestion import run_ingestion
from app.memory.vector_store import VectorStore


class SchedulerService:
    def __init__(self):
        self.store = VectorStore()
        self.assistant = CLIAssistant(self.store)
        self.scheduler = BackgroundScheduler()

    def run_ingestion_job(self):
        print("Running scheduled ingestion...")
        run_ingestion(store=self.store)

    def wake_up_job(self):
        print("Scheduled wake-up check starting...")
        self.assistant.wake_up_job()

    def start(self):
        # change these times if you want
        self.scheduler.add_job(
            self.run_ingestion_job,
            trigger="cron",
            hour=6,
            minute=0,
            id="daily_ingestion",
        )
        self.scheduler.add_job(
            self.wake_up_job,
            trigger="cron",
            hour=7,
            minute=10,
            id="morning_wake",
        )
        self.scheduler.add_job(
            self.wake_up_job,
            trigger="cron",
            hour=20,
            minute=0,
            id="evening_wake",
        )

        self.scheduler.start()
        print("Scheduler started. Waiting for jobs...")

        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("Shutting down scheduler...")
            self.scheduler.shutdown()


if __name__ == "__main__":
    SchedulerService().start()