"""Keep synchronous work owned until it finishes, even after caller cancellation."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import partial


class WorkOwnerClosed(RuntimeError):
    """The application has stopped admitting synchronous work."""


_current_owner = ContextVar("idm_work_owner", default=None)


class WorkOwner:
    """Track actual executor futures for one application lifespan."""

    def __init__(self):
        self.closing = False
        self._workers = set()

    @contextmanager
    def bind(self):
        token = _current_owner.set(self)
        try:
            yield
        finally:
            _current_owner.reset(token)

    def submit(self, call):
        # There is no await between admission and registration on the event loop.
        if self.closing:
            raise WorkOwnerClosed("Service is shutting down; retry later")
        worker = asyncio.get_running_loop().run_in_executor(None, call)
        self._workers.add(worker)
        worker.add_done_callback(self._finished)
        return worker

    def _finished(self, worker):
        self._workers.discard(worker)
        if not worker.cancelled():
            worker.exception()  # Retrieve failures without logging private details.

    async def drain(self):
        self.closing = True
        cancelled = False
        while self._workers:
            # return_exceptions retrieves failures even when a request itself
            # has disappeared. Its caller still receives its original outcome.
            pending = asyncio.gather(*self._workers, return_exceptions=True)
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    cancelled = True
            pending.result()
        if cancelled:
            raise asyncio.CancelledError


async def run_owned_sync(factory, *, on_cancel=None, **kwargs):
    # Cancelling a coroutine cannot stop its OS thread. Shield the worker so
    # callers retain their admission/model gates until the thread really exits.
    # A to_thread Task can itself be cancelled by asyncio's global shutdown,
    # even behind shield. An executor Future is not in all_tasks(); shielding
    # it keeps its completion tied to the actual thread. Preserve to_thread's
    # context propagation, including the application owner.
    call = partial(copy_context().run, partial(factory, **kwargs))
    owner = _current_owner.get()
    worker = owner.submit(call) if owner is not None else asyncio.get_running_loop().run_in_executor(None, call)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            try:
                result = worker.result()
            except Exception:
                pass  # The worker is responsible for its own failure cleanup.
            else:
                if on_cancel is not None:
                    on_cancel(result)
        raise
