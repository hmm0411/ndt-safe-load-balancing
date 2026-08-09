from enum import Enum

class ControllerRole(str, Enum):
    NOCHANGE = "NOCHANGE"
    EQUAL = "EQUAL"
    MASTER = "MASTER"
    SLAVE = "SLAVE"

class TransactionState(str, Enum):
    PREPARING = "PREPARING"
    ROLE_SWITCHING = "ROLE_SWITCHING"
    VERIFYING = "VERIFYING"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"
    ROLLING_BACK = "ROLLING_BACK"
    RESTORED = "RESTORED"
