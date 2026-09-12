import sys
import uuid
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import defaultdict
import threading
from queue import Queue, Empty


class EventType(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONTENT_PLAY = "content_play"
    CONTENT_PAUSE = "content_pause"
    CONTENT_RESUME = "content_resume"
    CONTENT_COMPLETE = "content_complete"
    CONTENT_SKIP = "content_skip"
    CONTENT_SEEK = "content_seek"
    RECOMMENDATION_IMPRESSION = "recommendation_impression"
    RECOMMENDATION_CLICK = "recommendation_click"
    RECOMMENDATION_DISMISS = "recommendation_dismiss"
    SEARCH_QUERY = "search_query"
    SEARCH_RESULT_CLICK = "search_result_click"
    SEARCH_NO_RESULTS = "search_no_results"
    FILTER_APPLIED = "filter_applied"
    RATING_SUBMITTED = "rating_submitted"
    WATCHLIST_ADD = "watchlist_add"
    WATCHLIST_REMOVE = "watchlist_remove"
    SHARE_CONTENT = "share_content"
    DOWNLOAD_START = "download_start"
    DOWNLOAD_COMPLETE = "download_complete"
    ERROR_OCCURRED = "error_occurred"
    SUBSCRIPTION_CHANGE = "subscription_change"
    DEVICE_REGISTERED = "device_registered"


@dataclass
class AnalyticsEvent:
    event_id: str
    event_type: EventType
    partner_id: str
    user_id: str
    session_id: Optional[str]
    timestamp: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalyticsEvent":
        data = data.copy()
        data["event_type"] = EventType(data["event_type"])
        return cls(**data)


class EventBatch:
    def __init__(self, events: List[AnalyticsEvent], batch_id: str = None):
        self.batch_id = batch_id or str(uuid.uuid4())
        self.events = events
        self.created_at = datetime.utcnow().isoformat()
        self.size = len(events)

    def to_json_lines(self) -> str:
        return "\n".join(e.to_json() for e in self.events)


class EventIngestion:
    def __init__(
        self,
        batch_size: int = 100,
        flush_interval_seconds: int = 30,
        max_queue_size: int = 10000
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self.max_queue_size = max_queue_size
        self.queue: Queue = Queue(maxsize=max_queue_size)
        self.handlers: List[Callable[[EventBatch], None]] = []
        self.running = False
        self._flush_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.stats = {
            "events_received": 0,
            "events_flushed": 0,
            "events_dropped": 0,
            "batches_sent": 0,
            "errors": 0
        }

    def add_handler(self, handler: Callable[[EventBatch], None]):
        with self._lock:
            self.handlers.append(handler)

    def ingest(self, event: AnalyticsEvent) -> bool:
        self.stats["events_received"] += 1
        try:
            self.queue.put_nowait(event)
            return True
        except:
            self.stats["events_dropped"] += 1
            return False

    def ingest_batch(self, events: List[AnalyticsEvent]) -> int:
        count = 0
        for event in events:
            if self.ingest(event):
                count += 1
        return count

    def start(self):
        if self.running:
            return
        self.running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def stop(self, timeout: float = 10.0):
        self.running = False
        self._flush_queue()
        if self._flush_thread:
            self._flush_thread.join(timeout=timeout)

    def _flush_loop(self):
        last_flush = time.time()
        batch = []

        while self.running:
            try:
                event = self.queue.get(timeout=1.0)
                batch.append(event)

                if len(batch) >= self.batch_size:
                    self._send_batch(batch)
                    batch = []
                    last_flush = time.time()

            except Empty:
                if batch and (time.time() - last_flush) >= self.flush_interval:
                    self._send_batch(batch)
                    batch = []
                    last_flush = time.time()

        if batch:
            self._send_batch(batch)

    def _send_batch(self, events: List[AnalyticsEvent]):
        if not events:
            return

        batch = EventBatch(events)
        for handler in self.handlers:
            try:
                handler(batch)
            except Exception as e:
                self.stats["errors"] += 1
                print(f"Event handler error: {e}", file=sys.stderr)

        self.stats["events_flushed"] += len(events)
        self.stats["batches_sent"] += 1

    def _flush_queue(self):
        batch = []
        while not self.queue.empty():
            try:
                batch.append(self.queue.get_nowait())
            except Empty:
                break
        if batch:
            self._send_batch(batch)

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "queue_size": self.queue.qsize(),
            "running": self.running
        }


class ConsoleEventHandler:
    def __call__(self, batch: EventBatch):
        print(f"[Batch {batch.batch_id[:8]}] {batch.size} events", file=sys.stderr)


class FileEventHandler:
    def __init__(self, filepath: str, rotate_mb: int = 100):
        self.filepath = filepath
        self.rotate_mb = rotate_mb * 1024 * 1024
        self.current_size = 0
        self._ensure_file()

    def _ensure_file(self):
        import os
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if os.path.exists(self.filepath):
            self.current_size = os.path.getsize(self.filepath)

    def __call__(self, batch: EventBatch):
        with open(self.filepath, "a") as f:
            f.write(batch.to_json_lines() + "\n")
        self.current_size += len(batch.to_json_lines())
        if self.current_size > self.rotate_mb:
            self._rotate()

    def _rotate(self):
        import os
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        rotated = self.filepath.replace(".jsonl", f"_{timestamp}.jsonl")
        os.rename(self.filepath, rotated)
        self.current_size = 0


class KafkaEventHandler:
    def __init__(self, bootstrap_servers: str, topic: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self._init_producer()

    def _init_producer(self):
        try:
            from kafka import KafkaProducer
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
        except ImportError:
            print("kafka-python not installed, Kafka handler disabled", file=sys.stderr)

    def __call__(self, batch: EventBatch):
        if not self.producer:
            return
        for event in batch.events:
            self.producer.send(
                self.topic,
                key=event.user_id,
                value=event.to_dict()
            )
        self.producer.flush()


class ClickHouseEventHandler:
    def __init__(self, host: str, port: int, database: str, table: str, user: str = "default", password: str = ""):
        self.host = host
        self.port = port
        self.database = database
        self.table = table
        self.client = None
        self._init_client(user, password)

    def _init_client(self, user: str, password: str):
        try:
            import clickhouse_connect
            self.client = clickhouse_connect.get_client(
                host=self.host, port=self.port,
                database=self.database, username=user, password=password
            )
        except ImportError:
            print("clickhouse-connect not installed, ClickHouse handler disabled", file=sys.stderr)

    def __call__(self, batch: EventBatch):
        if not self.client:
            return

        rows = []
        for event in batch.events:
            row = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "partner_id": event.partner_id,
                "user_id": event.user_id,
                "session_id": event.session_id or "",
                "timestamp": event.timestamp,
                "payload": json.dumps(event.payload),
                "metadata": json.dumps(event.metadata)
            }
            rows.append(row)

        self.client.insert(self.table, rows, column_names=list(rows[0].keys()))


class EventSchemaValidator:
    REQUIRED_FIELDS = {
        EventType.SESSION_START: ["device_type", "party_type"],
        EventType.SESSION_END: ["duration_minutes", "watch_events"],
        EventType.CONTENT_PLAY: ["content_id", "content_type", "position"],
        EventType.CONTENT_COMPLETE: ["content_id", "watch_time_minutes", "completed"],
        EventType.CONTENT_SKIP: ["content_id", "skip_position", "watch_time_before_skip"],
        EventType.RECOMMENDATION_IMPRESSION: ["recommendation_ids", "positions"],
        EventType.RECOMMENDATION_CLICK: ["recommendation_id", "position"],
        EventType.SEARCH_QUERY: ["query", "results_count"],
        EventType.RATING_SUBMITTED: ["content_id", "rating"],
        EventType.SUBSCRIPTION_CHANGE: ["old_tier", "new_tier"]
    }

    @classmethod
    def validate(cls, event: AnalyticsEvent) -> Tuple[bool, List[str]]:
        required = cls.REQUIRED_FIELDS.get(event.event_type, [])
        missing = [f for f in required if f not in event.payload]
        return len(missing) == 0, missing


def create_ingestion_pipeline(
    output_path: str = "data/analytics/events.jsonl",
    use_kafka: bool = False,
    kafka_servers: str = "",
    kafka_topic: str = "",
    use_clickhouse: bool = False,
    clickhouse_config: Dict = None
) -> EventIngestion:
    ingestion = EventIngestion(batch_size=100, flush_interval_seconds=30)
    ingestion.add_handler(ConsoleEventHandler())
    ingestion.add_handler(FileEventHandler(output_path))

    if use_kafka and kafka_servers:
        ingestion.add_handler(KafkaEventHandler(kafka_servers, kafka_topic))

    if use_clickhouse and clickhouse_config:
        ingestion.add_handler(ClickHouseEventHandler(**clickhouse_config))

    return ingestion


def build_event(
    event_type: EventType,
    partner_id: str,
    user_id: str,
    session_id: Optional[str],
    payload: Dict[str, Any],
    metadata: Dict[str, Any] = None
) -> AnalyticsEvent:
    return AnalyticsEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        partner_id=partner_id,
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.utcnow().isoformat(),
        payload=payload,
        metadata=metadata or {}
    )


def main():
    ingestion = create_ingestion_pipeline("data/analytics/events.jsonl")
    ingestion.start()

    events = [
        build_event(
            EventType.SESSION_START, "airline_default", "user_123", "sess_456",
            {"device_type": "seatback", "party_type": "solo", "flight_number": "AA123"}
        ),
        build_event(
            EventType.CONTENT_PLAY, "airline_default", "user_123", "sess_456",
            {"content_id": 550, "content_type": "movie", "position": 0}
        ),
        build_event(
            EventType.RECOMMENDATION_IMPRESSION, "airline_default", "user_123", "sess_456",
            {"recommendation_ids": [550, 680, 13], "positions": [0, 1, 2]}
        ),
        build_event(
            EventType.RECOMMENDATION_CLICK, "airline_default", "user_123", "sess_456",
            {"recommendation_id": 550, "position": 0}
        ),
        build_event(
            EventType.CONTENT_COMPLETE, "airline_default", "user_123", "sess_456",
            {"content_id": 550, "watch_time_minutes": 136, "completed": True}
        ),
        build_event(
            EventType.SESSION_END, "airline_default", "user_123", "sess_456",
            {"duration_minutes": 150, "watch_events": [{"content_id": 550, "watch_time_minutes": 136}]}
        )
    ]

    ingestion.ingest_batch(events)
    time.sleep(1)
    ingestion.stop()

    print(f"Stats: {ingestion.get_stats()}", file=sys.stderr)


if __name__ == "__main__":
    main()