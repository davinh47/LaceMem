from prompts.raw_memory_manager import RawMemoryManagerPrompt
from prompts.index_manager import IndexManagerPrompt

from men_llm.llm_client import LLMClient

import json
import math

import uuid
from db.db_index import normalize_fingerprint
from db.db_edge import insert_edge
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

class Manager:
    """
    Base class for all manager types.
    """
    def __init__(
        self,
        model_name="gpt-4o",
        api_key_path="./api.key",
    ):
        self.llm = LLMClient(api_key_path=api_key_path, default_model=model_name)

class RawMemoryManager(Manager):
    def __init__(
        self,
        model_name="gpt-4o",
    ):
        super().__init__(model_name)
        # Pending conversational focus (kept in-memory only; not stored in DB)
        # Separate topic vs time so they can expire independently.
        self._pending_ctx: Dict[str, Any] = dict(
            topic=None,      # Optional[str]
            time=None,       # Optional[str]
            topic_age=0,     # int (turns since last set)
            time_age=0,      # int (turns since last set)
        )

    @staticmethod
    def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
        """Normalize tool call return shapes into a list of {name, arguments}."""
        if not tool_calls:
            return []
        if isinstance(tool_calls, list):
            return tool_calls
        if isinstance(tool_calls, dict):
            maybe = tool_calls.get("tool_calls") or tool_calls.get("tool_call")
            if isinstance(maybe, list):
                return maybe
            if isinstance(maybe, dict):
                return [maybe]
        return []

    @staticmethod
    def _coerce_args(args: Any) -> Dict[str, Any]:
        """Coerce tool-call arguments into a dict.

        Accepts:
          - dict
          - JSON string representing a dict
          - plain string (treated as context_text)
        """
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            s = args.strip()
            if not s:
                return {}
            # Try JSON first
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    j = json.loads(s)
                    if isinstance(j, dict):
                        return j
                except Exception:
                    pass
            # Fallback: treat as a raw context string
            return {"context_text": s}
        return {}

    # --- pending context (topic/time) management ---

    def pending_topic_set(self, topic_text: str) -> None:
        text = (topic_text or "").strip()
        if not text:
            return
        self._pending_ctx["topic"] = text
        self._pending_ctx["topic_age"] = 0

    def pending_time_set(self, time_text: str) -> None:
        text = (time_text or "").strip()
        if not text:
            return
        self._pending_ctx["time"] = text
        self._pending_ctx["time_age"] = 0

    def pending_topic_clear(self) -> None:
        self._pending_ctx["topic"] = None
        self._pending_ctx["topic_age"] = 0

    def pending_time_clear(self) -> None:
        self._pending_ctx["time"] = None
        self._pending_ctx["time_age"] = 0

    def _age_pending_ctx(self) -> None:
        """Age pending topic/time once per turn and expire each independently."""
        # Topic
        if self._pending_ctx.get("topic") is not None:
            self._pending_ctx["topic_age"] = int(self._pending_ctx.get("topic_age") or 0) + 1
            if self._pending_ctx["topic_age"] > 4:
                self.pending_topic_clear()
        else:
            self._pending_ctx["topic_age"] = 0

        # Time
        if self._pending_ctx.get("time") is not None:
            self._pending_ctx["time_age"] = int(self._pending_ctx.get("time_age") or 0) + 1
            if self._pending_ctx["time_age"] > 4:
                self.pending_time_clear()
        else:
            self._pending_ctx["time_age"] = 0

    def __call__(
        self,
        *,
        conn,
        dia_id: str,
        speaker: str,
        user_text: str,
        session_time: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Context-aware raw ingest:
        - Injects pending topic/time into the prompt so short or coreferential turns can be rewritten.
        - LLM may call pending_* tools, then optionally raw_memory_insert.
        - Returns the inserted row dict, or None if nothing stored this turn.
        """
        rt = session_time if isinstance(session_time, datetime) else datetime.now(timezone.utc)
        uid = "rm_" + uuid.uuid4().hex[:8]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "raw_memory_insert",
                    "description": "Insert one raw memory row into the raw_memory table and return the full inserted row.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "unique_id": {"type": "string", "description": "Unique raw memory id, e.g. rm_ab12cd34"},
                            "dia_id": {"type": "string", "description": "Dialogue/session id"},
                            "speaker": {"type": "string", "description": "Speaker, e.g. user/assistant"},
                            "raw_text": {"type": "string", "description": "Memory text to store (may be normalized/cleaned; remove filler/greetings/questions when possible)."},
                            "record_time": {"type": "string", "description": "ISO timestamp (timezone-aware)"},
                        },
                        "required": ["unique_id", "dia_id", "speaker", "raw_text", "record_time"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "pending_topic_set",
                    "description": "Set/update the pending TOPIC focus (e.g., who/what the conversation is about).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic_text": {"type": "string", "description": "Short topic label to carry forward."}
                        },
                        "required": ["topic_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "pending_time_set",
                    "description": "Set/update the pending TIME focus (e.g., 'this summer', 'next month', 'last Saturday').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time_text": {"type": "string", "description": "Short time phrase to carry forward."}
                        },
                        "required": ["time_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "pending_time_clear",
                    "description": "Clear the pending TIME focus when it is no longer relevant.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "pending_topic_clear",
                    "description": "Clear the pending TOPIC focus when it is no longer relevant.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

        # Age pending topic/time once per new turn
        self._age_pending_ctx()

        print(
            f"[RAW][CTX_IN ] dia_id={dia_id} speaker={speaker} "
            f"topic_age={self._pending_ctx.get('topic_age')} topic={self._pending_ctx.get('topic')} "
            f"time_age={self._pending_ctx.get('time_age')} time={self._pending_ctx.get('time')}"
        )

        messages = self._build_messages(
            dia_id=dia_id,
            speaker=speaker,
            user_text=user_text,
            record_time=rt,
            unique_id=uid,
        )

        # Ask the model to decide whether to store by optionally calling the tool.
        tool_calls = self.llm.call_with_tools_messages(messages=messages, tools=tools)

        calls = self._normalize_tool_calls(tool_calls)
        if not calls:
            return None

        # print(f"[RAW][TOOL_CALLS] {calls}")

        inserted_row: Optional[Dict[str, Any]] = None

        # Process calls in order; allow context update + optional raw insert in the same turn.
        for call in calls:
            name = call.get("name")
            args = self._coerce_args(call.get("arguments"))

            if name == "pending_topic_set":
                t = args.get("topic_text")
                if t is None:
                    t = args.get("text") or args.get("topic")
                if isinstance(t, str) and t.strip():
                    self.pending_topic_set(t)
                continue

            if name == "pending_time_set":
                t = args.get("time_text")
                if t is None:
                    t = args.get("text") or args.get("time")
                if isinstance(t, str) and t.strip():
                    self.pending_time_set(t)
                continue

            if name == "pending_topic_clear":
                self.pending_topic_clear()
                continue

            if name == "pending_time_clear":
                self.pending_time_clear()
                continue

            if name != "raw_memory_insert":
                continue

            # Allow the model to store a *normalized* / cleaned memory text.
            cleaned_text = args.get("raw_text")
            if isinstance(cleaned_text, str):
                cleaned_text = cleaned_text.strip()
            else:
                cleaned_text = ""

            # Fallback: if the model didn't provide a cleaned version, store the original.
            if not cleaned_text:
                cleaned_text = user_text

            inserted_row = self.raw_memory_insert(
                conn=conn,
                unique_id=uid,
                dia_id=dia_id,
                speaker=speaker,
                raw_text=cleaned_text,
                record_time=rt,
            )

        return inserted_row

    def _build_messages(
        self,
        *,
        dia_id: str,
        speaker: str,
        user_text: str,
        record_time: datetime,
        unique_id: str,
    ) -> list[dict[str, str]]:
        """
        Build MIRIX-style messages: prompt is rules, input is injected as message context.
        """
        system_prompt = RawMemoryManagerPrompt

        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": "=== Manager Input (runtime injected) ==="},
            {"role": "system", "content": f"dia_id: {dia_id}"},
            {"role": "system", "content": f"speaker: {speaker}"},
            {"role": "system", "content": f"record_time: {record_time.isoformat()}"},
            {"role": "system", "content": f"unique_id: {unique_id}"},
            {"role": "system", "content": f"pending_topic: {self._pending_ctx.get('topic')}"},
            {"role": "system", "content": f"pending_time: {self._pending_ctx.get('time')}"},
            {"role": "system", "content": f"pending_topic_age: {self._pending_ctx.get('topic_age')}"},
            {"role": "system", "content": f"pending_time_age: {self._pending_ctx.get('time_age')}"},
            {"role": "user", "content": user_text},
            {
                "role": "system",
                "content": (
                    "You may call tools to manage memory.\n"
                    "- Use pending_topic_set to set/update the current conversation TOPIC focus.\n"
                    "- Use pending_time_set to set/update the current conversation TIME focus.\n"
                    "- Use pending_time_clear only when that focus is no longer relevant.\n"
                    "- Use raw_memory_insert only if the current message contains information worth remembering.\n"
                    "When calling tools: do not invent facts. If nothing needs storing or updating, call no tools.\n"
                ),
            },
        ]

    def raw_memory_insert(
        self,
        *,
        conn,
        unique_id: str,
        dia_id: str,
        speaker: str,
        raw_text: str,
        record_time: datetime,
    ) -> Dict[str, Any]:
        """
        Insert a new raw memory row.

        Parameters:
            conn: A PostgreSQL DB-API connection (cursor() + execute() + commit()).
            unique_id: Unique raw memory id (generated by the manager, e.g. "rm_xxxxxxxx").
            dia_id: Dialogue/session group id.
            speaker: Who said it (e.g., "user" / "assistant").
            raw_text: The original text to store verbatim.
            record_time: System record time (timezone-aware).

        Returns:
            The full inserted row as a dict with keys:
            id, dia_id, speaker, raw_text, record_time, update_time
        """
        uid = unique_id
        rt = record_time

        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO raw_memory (id, dia_id, speaker, raw_text, record_time, update_time)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, dia_id, speaker, raw_text, record_time, update_time
            """,
            (uid, dia_id, speaker, raw_text, rt, rt),
        )
        row = cur.fetchone()
        conn.commit()

        if not row:
            raise RuntimeError("raw_memory_insert failed: no row returned")

        cols = ["id", "dia_id", "speaker", "raw_text", "record_time", "update_time"]
        return {cols[i]: row[i] for i in range(len(cols))}

    def raw_memory_update(
        self,
        *,
        conn,
        unique_id: str,
        new_raw_text: str,
        update_time: datetime,
    ) -> Dict[str, Any]:
        """
        Update an existing raw memory row's text and update_time.

        Parameters:
            conn: A PostgreSQL DB-API connection.
            unique_id: The raw_memory.id to update.
            new_raw_text: New text to replace raw_text.
            update_time: System update time (timezone-aware, REQUIRED).

        Returns:
            The full updated row as a dict with keys:
            id, dia_id, speaker, raw_text, record_time, update_time

        Raises:
            KeyError if the row does not exist.
        """
        ut = update_time

        cur = conn.cursor()
        cur.execute(
            """
            UPDATE raw_memory
            SET raw_text = %s,
                update_time = %s
            WHERE id = %s
            RETURNING id, dia_id, speaker, raw_text, record_time, update_time
            """,
            (new_raw_text, ut, unique_id),
        )
        row = cur.fetchone()
        conn.commit()

        if not row:
            raise KeyError(f"raw_memory_update: id not found: {unique_id}")

        cols = ["id", "dia_id", "speaker", "raw_text", "record_time", "update_time"]
        return {cols[i]: row[i] for i in range(len(cols))}

    def raw_memory_delete(self, *, conn, unique_id: str) -> str:
        """
        Delete one raw memory row.

        Parameters:
            conn: A PostgreSQL DB-API connection.
            unique_id: The raw_memory.id to delete.

        Returns:
            The deleted memory id.

        Raises:
            KeyError if the row does not exist.
        """
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM raw_memory
            WHERE id = %s
            RETURNING id
            """,
            (unique_id,),
        )
        row = cur.fetchone()
        conn.commit()

        if not row:
            raise KeyError(f"raw_memory_delete: id not found: {unique_id}")

        return str(row[0])


class IndexManager(Manager):
    def __init__(
        self,
        model_name="gpt-4o",
    ):
        super().__init__(model_name)

    def __call__(
        self,
        *,
        conn,
        raw_row: Dict[str, Any],
    ) -> Optional[list[Dict[str, Any]]]:
        """
        Insert-only IndexManager (SPO-oriented).

        Input:
          raw_row: dict returned by RawMemoryManager.raw_memory_insert with keys:
            id, dia_id, speaker, raw_text, record_time, update_time

        Behavior:
          - Ask LLM to extract 0..N SPO triples (+ optional event_time) from raw_text.
          - For each extracted triple, insert one row into memory_index.
          - Return list of inserted index rows, or None if nothing to index.

        Notes:
          - We pin raw_id/dia_id/timestamps/group_id in Python.
          - record_time == raw_row['record_time'] (record time).
          - event_time may be resolved (ISO) or left null with event_time_text.
          - memory_type is inherited directly from raw_row['memory_type'].
        """
        raw_id = str(raw_row["id"])
        dia_id = str(raw_row["dia_id"])
        speaker = str(raw_row.get("speaker", "user"))
        raw_text = str(raw_row.get("raw_text", ""))
        record_time: datetime = raw_row["record_time"]
        update_time: datetime = raw_row.get("update_time", record_time)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "memory_index_insert",
                    "description": (
                        "Insert ONE SPO triple (subject, predicate, object) derived from the current raw message. "
                        "Call this multiple times if multiple triples exist. "
                        "Use event_time if the message contains a resolvable time; otherwise use event_time_text."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_type": {
                                "type": "string",
                                "description": "Memory category for this triple (core/episodic/procedural/resource/knowledge_vault/semantic/other).",
                            },
                            "subject": {
                                "type": "string",
                                "description": "Subject string (e.g., Jack / Unity / user / report).",
                            },
                            "subject_type": {
                                "type": "string",
                                "description": "Type of subject (person/concept/event/resource/procedure/user/org/place/etc.).",
                            },
                            "predicate": {
                                "type": "string",
                                "description": "Relation/action (e.g., is_a / lives_in / met / had_coffee_with / works_at / happened_at / has_step).",
                            },
                            "object": {
                                "type": "string",
                                "description": "Object/value string (e.g., game engine / Shanghai / Mary / 2025-12-11).",
                            },
                            "object_type": {
                                "type": "string",
                                "description": "Type of object (person/place/category/time/org/value/text/etc.).",
                            },
                            "event_time": {
                                "type": "string",
                                "description": (
                                    "Event time as ISO-8601 timestamp if you can resolve it using record_time as the reference. "
                                    "If you cannot resolve it precisely, omit this and use event_time_text."
                                ),
                            },
                            "event_time_text": {
                                "type": "string",
                                "description": "Original time phrase when event_time cannot be resolved (e.g., 'ten-ish days ago').",
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Optional extraction confidence between 0 and 1.",
                            },
                        },
                        "required": ["subject", "predicate"],
                    },
                },
            }
        ]

        messages = self._build_messages(
            raw_id=raw_id,
            dia_id=dia_id,
            speaker=speaker,
            raw_text=raw_text,
            record_time=record_time,
        )

        tool_calls = self.llm.call_with_tools_messages(messages=messages, tools=tools)
        # print(f"[DEBUG][IndexManager] raw_id={raw_id} tool_calls={tool_calls}")
        if not tool_calls:
            return None

        # Normalize tool_calls to a list
        calls: list[Dict[str, Any]] = []
        if isinstance(tool_calls, list):
            calls = tool_calls
        elif isinstance(tool_calls, dict):
            maybe = tool_calls.get("tool_calls") or tool_calls.get("tool_call")
            if isinstance(maybe, list):
                calls = maybe
            elif isinstance(maybe, dict):
                calls = [maybe]

        inserted_rows: list[Dict[str, Any]] = []

        for call in calls:
            if call.get("name") != "memory_index_insert":
                continue
            args = call.get("arguments") or {}

            memory_type = (args.get("memory_type") or None)
            if isinstance(memory_type, str):
                memory_type = memory_type.strip() or None
            if not memory_type:
                memory_type = "other"

            subject = (args.get("subject") or "").strip()
            predicate = (args.get("predicate") or "").strip()
            if not subject or not predicate:
                continue

            subject_type = (args.get("subject_type") or None)
            if isinstance(subject_type, str):
                subject_type = subject_type.strip() or None

            obj = (args.get("object") or None)
            if isinstance(obj, str):
                obj = obj.strip() or None

            object_type = (args.get("object_type") or None)
            if isinstance(object_type, str):
                object_type = object_type.strip() or None

            # Keep event_time as TEXT (do not parse to datetime)
            event_time = None
            event_time_str = args.get("event_time")
            if isinstance(event_time_str, str):
                event_time = event_time_str.strip() or None

            event_time_text = args.get("event_time_text") or None
            if isinstance(event_time_text, str):
                event_time_text = event_time_text.strip() or None

            confidence = args.get("confidence")
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except Exception:
                    confidence = None

            fingerprint = normalize_fingerprint(
                subject=subject,
                predicate=predicate,
                object=obj,
                event_time=event_time,
            )

            row = self.memory_index_insert(
                conn=conn,
                raw_id=raw_id,
                dia_id=dia_id,
                speaker=speaker,
                memory_type=memory_type,
                subject=subject,
                subject_type=subject_type,
                predicate=predicate,
                object_=obj,
                object_type=object_type,
                event_time=event_time,
                event_time_text=event_time_text,
                fingerprint=fingerprint,
                confidence=confidence,
                record_time=record_time,
                updated_at=update_time,
            )
            inserted_rows.append(row)

        return inserted_rows or None

    def _build_messages(
        self,
        *,
        raw_id: str,
        dia_id: str,
        speaker: str,
        raw_text: str,
        record_time: datetime,
    ) -> list[dict[str, str]]:
        system_prompt = IndexManagerPrompt

        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": "=== Manager Input (runtime injected) ==="},
            {"role": "system", "content": f"raw_id: {raw_id}"},
            {"role": "system", "content": f"dia_id: {dia_id}"},
            {"role": "system", "content": f"speaker: {speaker}"},
            {"role": "system", "content": f"record_time: {record_time.isoformat()}"},
            {"role": "user", "content": raw_text},
            {
                "role": "system",
                "content": (
                    "Extract 0..N SPO triples from the user message. "
                    "Call memory_index_insert once per triple. "
                    "If nothing worth indexing, do not call any tool."
                ),
            },
        ]

    def memory_index_insert(
        self,
        *,
        conn,
        raw_id: str,
        dia_id: str,
        speaker: str,
        memory_type: str,
        subject: str,
        subject_type: Optional[str],
        predicate: str,
        object_: Optional[str],
        object_type: Optional[str],
        event_time: Optional[str],
        event_time_text: Optional[str],
        fingerprint: Optional[str],
        confidence: Optional[float],
        record_time: datetime,
        updated_at: datetime,
    ) -> Dict[str, Any]:
        """Insert one row into memory_index and return the inserted row dict."""
        idx_id = "mi_" + uuid.uuid4().hex[:8]

        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO memory_index (
              id, raw_id, dia_id, speaker,
              memory_type,
              subject, subject_type,
              predicate,
              object, object_type,
              event_time, event_time_text,
              fingerprint, confidence,
              record_time, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING
              id, raw_id, dia_id, speaker,
              memory_type,
              subject, subject_type,
              predicate,
              object, object_type,
              event_time, event_time_text,
              fingerprint, confidence,
              record_time, updated_at
            """,
            (
                idx_id, raw_id, dia_id, speaker,
                memory_type,
                subject, subject_type,
                predicate,
                object_, object_type,
                event_time, event_time_text,
                fingerprint, confidence,
                record_time, updated_at,
            ),
        )
        row = cur.fetchone()
        conn.commit()

        if not row:
            raise RuntimeError("memory_index_insert failed: no row returned")

        cols = [
            "id", "raw_id", "dia_id", "speaker",
            "memory_type",
            "subject", "subject_type",
            "predicate",
            "object", "object_type",
            "event_time", "event_time_text",
            "fingerprint", "confidence",
            "record_time", "updated_at",
        ]
        return {cols[i]: row[i] for i in range(len(cols))}

class LinkManager(Manager):
    """Build edges between `memory_index` nodes.

    Design goals (v1):
      - No LLM in linking. The index is already structured.
      - Deterministic edges first (same_raw, same_dia).
      - Optional clustering edges using TEXT embeddings (semantic_sim).

    This manager expects the DB schema:
      - memory_index(id, raw_id, dia_id, speaker, subject, predicate, object, ...)
      - memory_index_edge(id, src_id, dst_id, edge_type, weight)

    Notes:
      - Undirected edges: db_edge.insert_edge should normalize (src_id, dst_id).
      - Weight meanings:
          same_raw: 1.0
          same_dia: 0.7 (tunable)
          semantic_sim: cosine similarity in [0, 1]
    """

    def __init__(
        self,
        model_name="gpt-4o",
        *,
        semantic_threshold: float = 0.7,
        max_candidates: int = 200,
    ):
        super().__init__(model_name)
        self.semantic_threshold = float(semantic_threshold)
        self.max_candidates = int(max_candidates)

    def __call__(
        self,
        *,
        conn,
        new_index_rows: List[Dict[str, Any]],
        enable_clustering: bool = True,
    ) -> Dict[str, Any]:
        """Link newly inserted index rows.

        Args:
            conn: psycopg connection
            new_index_rows: rows returned by IndexManager (list of dicts)
            enable_clustering: if True, also build semantic_sim edges

        Returns:
            dict with counts.
        """
        if not new_index_rows:
            return {"edges_created": 0, "same_raw": 0, "same_dia": 0, "semantic_sim": 0}

        created_total = 0
        created_same_raw = 0
        created_same_dia = 0
        created_sem = 0

        # 1) same_raw edges (within the same raw_id)
        # We do it for each new node, but use INSERT ... ON CONFLICT at edge layer to dedupe.
        for row in new_index_rows:
            c = self._link_same_raw(conn=conn, node=row)
            created_total += c
            created_same_raw += c

        # 2) same_dia edges (within same dia_id, different raw_id)
        for row in new_index_rows:
            c = self._link_same_dia(conn=conn, node=row)
            created_total += c
            created_same_dia += c

        # 3) optional clustering edges
        if enable_clustering:
            for row in new_index_rows:
                c = self._link_semantic(conn=conn, node=row)
                created_total += c
                created_sem += c

        return {
            "edges_created": created_total,
            "same_raw": created_same_raw,
            "same_dia": created_same_dia,
            "semantic_sim": created_sem,
        }

    # -------------------------
    # Deterministic linking
    # -------------------------

    def _link_same_raw(self, *, conn, node: Dict[str, Any]) -> int:
        raw_id = str(node.get("raw_id") or "")
        node_id = str(node.get("id") or "")
        if not raw_id or not node_id:
            return 0

        cur = conn.cursor()
        cur.execute(
            """
            SELECT id
            FROM memory_index
            WHERE raw_id = %s
            """,
            (raw_id,),
        )
        rows = cur.fetchall() or []

        created = 0
        for (other_id,) in rows:
            other_id = str(other_id)
            if not other_id or other_id == node_id:
                continue
            ok = insert_edge(
                conn=conn,
                edge_id="me_" + uuid.uuid4().hex[:10],
                src_id=node_id,
                dst_id=other_id,
                edge_type="same_raw",
                weight=1.0,
            )
            if ok:
                created += 1
        return created

    def _link_same_dia(self, *, conn, node: Dict[str, Any]) -> int:
        dia_id = str(node.get("dia_id") or "")
        node_id = str(node.get("id") or "")
        raw_id = str(node.get("raw_id") or "")
        if not dia_id or not node_id:
            return 0

        cur = conn.cursor()
        cur.execute(
            """
            SELECT id
            FROM memory_index
            WHERE dia_id = %s
              AND raw_id <> %s
            """,
            (dia_id, raw_id),
        )
        rows = cur.fetchall() or []

        created = 0
        for (other_id,) in rows:
            other_id = str(other_id)
            if not other_id or other_id == node_id:
                continue
            ok = insert_edge(
                conn=conn,
                edge_id="me_" + uuid.uuid4().hex[:10],
                src_id=node_id,
                dst_id=other_id,
                edge_type="same_dia",
                weight=0.7,
            )
            if ok:
                created += 1
        return created

    # -------------------------
    # Clustering / semantic linking (text embeddings)
    # -------------------------

    def _canonical_text(self, node: Dict[str, Any]) -> str:
        """Build a canonical text representation for TEXT embeddings.

        v1: just SPO.
        """
        s = (node.get("subject") or "").strip()
        p = (node.get("predicate") or "").strip()
        o = (node.get("object") or "").strip()
        if o:
            return f"{s} {p} {o}".strip()
        return f"{s} {p}".strip()

    def _candidate_query(self, *, speaker: str, subject: str, obj: str) -> Tuple[str, Tuple[Any, ...]]:
        """Blocking query: same speaker + subject/object overlap.

        This avoids requiring exact predicate equality (too strict).
        """
        sql = (
            "SELECT id, subject, predicate, object "
            "FROM memory_index "
            "WHERE speaker = %s "
            "  AND (subject = %s OR object = %s OR subject = %s OR object = %s) "
            "ORDER BY record_time DESC "
            "LIMIT %s"
        )
        params = (speaker, subject, subject, obj, obj, self.max_candidates)
        return sql, params

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts using the embedding backend in LLMClient."""
        return self.llm.embed_texts(texts)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:

        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(len(a)):
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(dot / (math.sqrt(na) * math.sqrt(nb)))

    def _link_semantic(self, *, conn, node: Dict[str, Any]) -> int:
        """Build semantic_sim edges using text embeddings + threshold.

        Blocking:
          - same speaker (hard)
          - subject/object overlap (soft)

        Similarity:
          cosine(embed(canonical_text(node)), embed(canonical_text(candidate)))
        """
        node_id = str(node.get("id") or "")
        speaker = str(node.get("speaker") or "")
        subject = str(node.get("subject") or "").strip()
        obj = str(node.get("object") or "").strip()

        if not node_id or not speaker or not subject:
            return 0

        sql, params = self._candidate_query(speaker=speaker, subject=subject, obj=obj)
        cur = conn.cursor()
        cur.execute(sql, params)
        candidates = cur.fetchall() or []

        # Remove self
        cand_rows: List[Tuple[str, str, str, str]] = []
        for r in candidates:
            cid = str(r[0])
            if not cid or cid == node_id:
                continue
            cand_rows.append((cid, str(r[1] or ""), str(r[2] or ""), str(r[3] or "")))

        if not cand_rows:
            return 0

        node_text = self._canonical_text(node)
        cand_texts = [f"{s} {p} {o}".strip() if o else f"{s} {p}".strip() for (_, s, p, o) in cand_rows]

        try:
            embs = self._embed_texts([node_text] + cand_texts)
        except NotImplementedError:
            return 0

        node_emb = embs[0]
        cand_embs = embs[1:]

        created = 0
        for i, (cid, _, _, _) in enumerate(cand_rows):
            sim = self._cosine(node_emb, cand_embs[i])
            if sim >= self.semantic_threshold:
                ok = insert_edge(
                    conn=conn,
                    edge_id="me_" + uuid.uuid4().hex[:10],
                    src_id=node_id,
                    dst_id=cid,
                    edge_type="semantic_sim",
                    weight=sim,
                )
                if ok:
                    created += 1
        return created
