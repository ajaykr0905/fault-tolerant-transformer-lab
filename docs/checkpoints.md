# Checkpoint writer operation

Checkpoint stores support cooperating writers on local POSIX filesystems
(macOS/Linux). Every `save_checkpoint` holds an exclusive advisory `flock`
through temporary cleanup, contract checks, generation allocation, publication,
and retention pruning. Different stores have independent locks.

The default acquisition timeout is 30 seconds. Set
`writer_lock_timeout=5.0` to wait at most five seconds for another writer.
`TimeoutError` means no checkpoint cleanup or publication was performed by that
call; an already-created store and its lock file can remain. The timeout does
not bound serialization or filesystem operations after acquisition.

`.writer.lock` is a persistent inode, not a stale-work marker. Never remove or
replace it while processes may use the store: doing so can create separate
locks for the same store. The OS releases ownership when its descriptor closes,
including on process death. Ordinary exceptions also release the lock. A later
writer performs the existing crash cleanup only after acquiring ownership.

Readers remain lock-free and use the committed `LATEST` record and integrity
checks. This lock prevents cooperating writers from deleting each other's
active files; it does not authenticate checkpoints, coordinate arbitrary
external file mutations, or provide distributed/network-filesystem guarantees.
Platforms without POSIX `flock` fail explicitly; Windows is not supported.

See [checkpoint trust requirements](checkpoint-security.md) and
[process-crash recovery evidence](process-recovery.md).
