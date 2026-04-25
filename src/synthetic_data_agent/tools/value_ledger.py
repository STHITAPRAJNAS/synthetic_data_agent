"""Synthetic Value Ledger — cross-table PII consistency.

Problem
-------
When the same entity (e.g. customer "John Smith") appears in multiple tables
(customers.name, orders.billing_name, support_tickets.reporter), each table's
PII handler would independently generate a different synthetic name.  The
resulting dataset loses relational coherence: the same real person maps to
different synthetic identities in different tables.

Solution
--------
The ``SyntheticValueLedger`` is a persistent lookup table keyed by:

    (pipeline_run_id, semantic_type, sha256(original_value + pipeline_salt)[:32])

On first encounter the ledger generates *and stores* a synthetic value.
On every subsequent encounter — even in a different table — the ledger returns
the **same** synthetic value without ever seeing the original again.

Privacy contract
-----------------
- The **original value is never stored** in the ledger.  Only the first 32 hex
  chars of ``sha256(original_value + pipeline_salt)`` are stored.
- The ``pipeline_salt`` is a random 32-byte secret generated once per pipeline
  run.  Without the salt, the hash cannot be reversed even by an adversary with
  a copy of the ledger.
- Different pipeline runs use different salts, so the same original value maps
  to *different* synthetic values in independently generated datasets.

Semantic types
--------------
Values are bucketed by semantic type so the same hash for two different column
types (e.g. "Smith" as a name vs. "Smith" as a company suffix) gets independent
synthetic values:

    person_name, company_name, email, phone, ssn, iban, card_pan,
    ip_address, address_street, address_city, address_zip,
    free_text, generic

DB schema
---------
Single table ``synthetic_value_ledger`` with a unique constraint on
``(pipeline_run_id, semantic_type, original_hash)``.  INSERT is done via
``ON CONFLICT DO NOTHING`` so concurrent writes are safe.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Callable

import structlog
from sqlalchemy import String, UniqueConstraint, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from ..config import get_settings

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class _LedgerBase(DeclarativeBase):
    pass


class _LedgerEntry(_LedgerBase):
    __tablename__ = "synthetic_value_ledger"
    __table_args__ = (
        UniqueConstraint(
            "pipeline_run_id", "semantic_type", "original_hash",
            name="uq_ledger_entity",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pipeline_run_id: Mapped[str] = mapped_column(String(36), index=True)
    semantic_type: Mapped[str] = mapped_column(String(64), index=True)
    original_hash: Mapped[str] = mapped_column(String(32))
    synthetic_value: Mapped[str] = mapped_column(String(4096))


# ---------------------------------------------------------------------------
# Semantic type constants
# ---------------------------------------------------------------------------

class SemanticType:
    PERSON_NAME = "person_name"
    COMPANY_NAME = "company_name"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    IBAN = "iban"
    CARD_PAN = "card_pan"
    IP_ADDRESS = "ip_address"
    ADDRESS_STREET = "address_street"
    ADDRESS_CITY = "address_city"
    ADDRESS_ZIP = "address_zip"
    FREE_TEXT = "free_text"
    GENERIC = "generic"


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

class SyntheticValueLedger:
    """Cross-table, privacy-safe synthetic value registry.

    Usage
    -----
    Instantiate once per pipeline run (or as a module-level singleton — the
    ``pipeline_run_id`` scope ensures isolation between runs).

        ledger = SyntheticValueLedger()
        await ledger.init_db()

        synthetic_name = await ledger.lookup_or_generate(
            pipeline_run_id="run-123",
            pipeline_salt="abc...xyz",
            semantic_type=SemanticType.PERSON_NAME,
            original_value="John Smith",
            generator_fn=generate_synthetic_name,
        )
    """

    def __init__(self, database_url: str | None = None) -> None:
        url = database_url or get_settings().database_url
        self._engine = create_async_engine(url, pool_pre_ping=True)
        self._session_factory = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._local_cache: dict[tuple[str, str, str], str] = {}

    async def init_db(self) -> None:
        """Create the ledger table if it does not exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(_LedgerBase.metadata.create_all)
        logger.info("value_ledger_ready")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def lookup_or_generate(
        self,
        pipeline_run_id: str,
        pipeline_salt: str,
        semantic_type: str,
        original_value: str,
        generator_fn: Callable[[], str],
    ) -> str:
        """Return a consistent synthetic value for *original_value*.

        If a synthetic value has been generated before for this
        ``(pipeline_run_id, semantic_type, hash(original_value + salt))``
        triple, that value is returned.  Otherwise a new value is generated,
        persisted, and returned.

        Args:
            pipeline_run_id: Unique ID for the current generation run.
            pipeline_salt: Per-run secret mixed into the hash.
            semantic_type: Semantic type constant (use ``SemanticType.*``).
            original_value: The real value to map (never stored).
            generator_fn: Zero-argument callable that produces one new
                synthetic value.

        Returns:
            Consistent synthetic string.
        """
        h = _hash(original_value, pipeline_salt)
        cache_key = (pipeline_run_id, semantic_type, h)

        # 1. In-memory cache (avoids DB round-trip for hot values)
        if cache_key in self._local_cache:
            return self._local_cache[cache_key]

        # 2. DB lookup
        async with self._session_factory() as session:
            row = await session.scalar(
                select(_LedgerEntry.synthetic_value).where(
                    _LedgerEntry.pipeline_run_id == pipeline_run_id,
                    _LedgerEntry.semantic_type == semantic_type,
                    _LedgerEntry.original_hash == h,
                )
            )

        if row is not None:
            self._local_cache[cache_key] = row
            return row

        # 3. Generate + persist (UPSERT to handle concurrent inserts safely)
        synthetic = generator_fn()
        async with self._session_factory() as session:
            stmt = (
                pg_insert(_LedgerEntry)
                .values(
                    pipeline_run_id=pipeline_run_id,
                    semantic_type=semantic_type,
                    original_hash=h,
                    synthetic_value=synthetic,
                )
                .on_conflict_do_nothing(
                    constraint="uq_ledger_entity"
                )
            )
            await session.execute(stmt)
            await session.commit()

            # Re-read in case another concurrent writer won the race
            final = await session.scalar(
                select(_LedgerEntry.synthetic_value).where(
                    _LedgerEntry.pipeline_run_id == pipeline_run_id,
                    _LedgerEntry.semantic_type == semantic_type,
                    _LedgerEntry.original_hash == h,
                )
            )
            result = final or synthetic

        self._local_cache[cache_key] = result
        return result

    async def bulk_lookup_or_generate(
        self,
        pipeline_run_id: str,
        pipeline_salt: str,
        semantic_type: str,
        original_values: list[str],
        generator_fn: Callable[[], str],
    ) -> list[str]:
        """Batch version of ``lookup_or_generate``.

        Uses a single SELECT for all cache misses, then parallel inserts.
        Preserves the order of *original_values* in the result list.

        Args:
            pipeline_run_id: Current pipeline run ID.
            pipeline_salt: Per-run secret.
            semantic_type: Semantic type for all values in this batch.
            original_values: Original values to map (order preserved).
            generator_fn: Callable producing one new synthetic value.

        Returns:
            List of synthetic values in the same order as *original_values*.
        """
        if not original_values:
            return []

        hashes = [_hash(v, pipeline_salt) for v in original_values]

        # Build result array, fill from cache first
        results: list[str | None] = [None] * len(original_values)
        miss_indices: list[int] = []
        miss_hashes: list[str] = []

        for i, h in enumerate(hashes):
            cached = self._local_cache.get((pipeline_run_id, semantic_type, h))
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_hashes.append(h)

        if not miss_indices:
            return results  # type: ignore[return-value]

        # Batch DB lookup for all misses
        async with self._session_factory() as session:
            rows = await session.execute(
                select(_LedgerEntry.original_hash, _LedgerEntry.synthetic_value).where(
                    _LedgerEntry.pipeline_run_id == pipeline_run_id,
                    _LedgerEntry.semantic_type == semantic_type,
                    _LedgerEntry.original_hash.in_(miss_hashes),
                )
            )
            found: dict[str, str] = {r[0]: r[1] for r in rows}

        # Still-missing values → generate + bulk insert
        truly_missing_indices = [i for i in miss_indices if hashes[i] not in found]
        new_entries: list[dict] = []
        for i in truly_missing_indices:
            synthetic = generator_fn()
            h = hashes[i]
            found[h] = synthetic
            new_entries.append({
                "pipeline_run_id": pipeline_run_id,
                "semantic_type": semantic_type,
                "original_hash": h,
                "synthetic_value": synthetic,
            })

        if new_entries:
            async with self._session_factory() as session:
                stmt = (
                    pg_insert(_LedgerEntry)
                    .values(new_entries)
                    .on_conflict_do_nothing(constraint="uq_ledger_entity")
                )
                await session.execute(stmt)
                await session.commit()

        # Assemble ordered results
        for i in miss_indices:
            h = hashes[i]
            v = found.get(h, generator_fn())  # last-resort fallback
            results[i] = v
            self._local_cache[(pipeline_run_id, semantic_type, h)] = v

        return results  # type: ignore[return-value]

    async def clear_run(self, pipeline_run_id: str) -> int:
        """Delete all ledger entries for a completed run (optional cleanup).

        Returns the number of rows deleted.
        """
        from sqlalchemy import delete
        async with self._session_factory() as session:
            result = await session.execute(
                delete(_LedgerEntry).where(
                    _LedgerEntry.pipeline_run_id == pipeline_run_id
                )
            )
            await session.commit()
            n = result.rowcount
        self._local_cache = {
            k: v for k, v in self._local_cache.items()
            if k[0] != pipeline_run_id
        }
        logger.info("ledger_run_cleared", pipeline_run_id=pipeline_run_id, rows=n)
        return n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(value: str, salt: str) -> str:
    """SHA-256 of (value + salt), first 32 hex chars."""
    raw = (value + salt).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def compute_entity_hashes(
    values: list,
    pipeline_salt: str,
) -> list[str]:
    """Hash a column of original values with the pipeline salt.

    Called by the *generator agent* (which has access to real data).
    The resulting hash list is passed to the PII handler so it can
    maintain consistent synthetic values without receiving real data.

    Args:
        values: Column values from the real DataFrame.
        pipeline_salt: Per-run secret from the GenerationPlan.

    Returns:
        List of 32-char hex strings in the same order as *values*.
    """
    return [_hash(str(v) if v is not None else "", pipeline_salt) for v in values]


def infer_semantic_type(col_name: str, pii_category: str) -> str:
    """Infer the ledger semantic type from column name and PII category.

    This determines which "bucket" the ledger uses, so columns that
    semantically hold the same kind of value (e.g. ``customer_name``,
    ``billing_name``, ``contact_name``) all share the same synthetic
    identity space.

    Args:
        col_name: Column name (e.g. ``"billing_name"``).
        pii_category: PIICategory string (e.g. ``"DIRECT_PII"``).

    Returns:
        SemanticType constant string.
    """
    lc = col_name.lower()

    if "ssn" in lc or "social_security" in lc or "national_id" in lc:
        return SemanticType.SSN
    if "email" in lc or "e_mail" in lc:
        return SemanticType.EMAIL
    if "phone" in lc or "mobile" in lc or "tel" in lc or "fax" in lc:
        return SemanticType.PHONE
    if "iban" in lc or "bank_account" in lc or "account_number" in lc:
        return SemanticType.IBAN
    if "card" in lc or "pan" in lc or "credit" in lc or "debit" in lc:
        return SemanticType.CARD_PAN
    if "ip" in lc or "ip_address" in lc:
        return SemanticType.IP_ADDRESS
    if "street" in lc or "address1" in lc or "address_line" in lc:
        return SemanticType.ADDRESS_STREET
    if "city" in lc or "town" in lc or "municipality" in lc:
        return SemanticType.ADDRESS_CITY
    if "zip" in lc or "postal" in lc or "postcode" in lc:
        return SemanticType.ADDRESS_ZIP
    if "company" in lc or "employer" in lc or "org" in lc or "organisation" in lc:
        return SemanticType.COMPANY_NAME
    if "description" in lc or "notes" in lc or "comment" in lc or "text" in lc:
        return SemanticType.FREE_TEXT
    if (
        "name" in lc
        or "first" in lc
        or "last" in lc
        or "full_name" in lc
        or "contact" in lc
        or "person" in lc
    ):
        return SemanticType.PERSON_NAME

    # Fallback based on PII category
    if pii_category in ("DIRECT_PII",):
        return SemanticType.PERSON_NAME
    if pii_category in ("FINANCIAL_PII",):
        return SemanticType.CARD_PAN

    return SemanticType.GENERIC
