from datetime import datetime, timezone
from threading import RLock
from src.common.enums import TransactionState
from src.schemas.transaction import MigrationTransaction, TransactionEvent

def utc_now():
    return datetime.now(timezone.utc)

class TransactionConflict(RuntimeError):
    pass

class TransactionManager:
    def __init__(self):
        self._transactions = {}
        self._locked_switches = set()
        self._lock = RLock()

    def create(self, switch_id: str, source_controller: str, target_controller: str):
        with self._lock:
            if switch_id in self._locked_switches:
                raise TransactionConflict(f"{switch_id} already has an active transaction")
            tx = MigrationTransaction(
                switch_id=switch_id,
                source_controller=source_controller,
                target_controller=target_controller,
                started_at=utc_now(),
            )
            tx.history.append(TransactionEvent(TransactionState.PREPARING, tx.started_at))
            self._transactions[tx.transaction_id] = tx
            self._locked_switches.add(switch_id)
            return tx

    def transition(self, tx, state: TransactionState, detail: str | None = None):
        with self._lock:
            tx.state = state
            tx.history.append(TransactionEvent(state, utc_now(), detail))

    def fail(self, tx, reason: str):
        tx.failure_reason = reason
        self.transition(tx, TransactionState.FAILED, reason)

    def finish(self, tx):
        with self._lock:
            tx.finished_at = utc_now()
            self._locked_switches.discard(tx.switch_id)

    def list_all(self):
        with self._lock:
            return [x.to_dict() for x in self._transactions.values()]

    def locked_switches(self):
        with self._lock:
            return sorted(self._locked_switches)
