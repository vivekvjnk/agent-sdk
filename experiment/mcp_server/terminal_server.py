import os
import asyncio
import time
from typing import Dict, Optional
import logging

from fastmcp import FastMCP, Context
from openhands.tools.terminal.impl import TerminalExecutor
from openhands.tools.terminal.definition import TerminalAction

from starlette.types import ASGIApp, Scope, Receive, Send

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("terminal_server")

# Global registry for sessions
class SessionTerminalRegistry:
    def __init__(self, idle_timeout: int = 300):
        self._sessions: Dict[str, TerminalExecutor] = {}
        self._last_access: Dict[str, float] = {}
        self.idle_timeout = idle_timeout

    def get_terminal(self, session_id: str) -> TerminalExecutor:
        self._last_access[session_id] = time.time()
        if session_id not in self._sessions:
            logger.info(f"Creating new terminal for session: {session_id}")
            self._sessions[session_id] = TerminalExecutor(working_dir=os.getcwd())
        return self._sessions[session_id]

    def remove_session(self, session_id: str):
        if session_id in self._sessions:
            logger.info(f"Removing terminal for session: {session_id}")
            terminal = self._sessions.pop(session_id)
            self._last_access.pop(session_id, None)
            # Force cleanup if possible
            try:
                # TerminalExecutor doesn't have a close() in some versions, 
                # but it usually cleans up on __del__. 
                del terminal
            except Exception as e:
                logger.error(f"Error cleaning up terminal for session {session_id}: {e}")

    async def reaper_loop(self):
        """Periodically clean up idle sessions."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            idle_sessions = [
                sid for sid, last in self._last_access.items()
                if now - last > self.idle_timeout
            ]
            for sid in idle_sessions:
                logger.info(f"Session {sid} idle for {self.idle_timeout}s, reaping...")
                self.remove_session(sid)

registry = SessionTerminalRegistry()

class SessionCleanupMiddleware:
    """Middleware to catch DELETE /mcp and clean up sessions."""
    def __init__(self, app: ASGIApp, registry: SessionTerminalRegistry):
        self.app = app
        self.registry = registry

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http" and scope["method"] == "DELETE" and scope["path"] == "/mcp":
            # Extract session ID from headers
            headers = dict(scope.get("headers", []))
            # Headers are bytes in ASGI scope
            session_id = None
            for k, v in headers.items():
                if k.lower() == b"mcp-session-id":
                    session_id = v.decode("utf-8")
                    break
            
            if session_id:
                logger.info(f"Intercepted DELETE for session {session_id}, cleaning up...")
                self.registry.remove_session(session_id)
        
        await self.app(scope, receive, send)

mcp = FastMCP("Terminal-Session-Server")

@mcp.tool()
async def run_bash_command(command: str, ctx: Context) -> str:
    """Run a bash command in the persistent terminal for this session."""
    session_id = ctx.session_id
    logger.info(f"Running command for session {session_id}: {command}")
    
    terminal = registry.get_terminal(session_id)
    # TerminalExecutor.run is synchronous
    loop = asyncio.get_running_loop()
    action = TerminalAction(command=command)
    result = await loop.run_in_executor(None, terminal, action)
    
    return result.text

# Set up lifespan for reaper loop
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastMCP):
    reaper_task = asyncio.create_task(registry.reaper_loop())
    try:
        yield
    finally:
        reaper_task.cancel()
        try:
            await reaper_task
        except asyncio.CancelledError:
            pass

mcp._lifespan = lifespan

if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8801"))
    # Use http_app() to get the Starlette app
    # We specify transport="streamable-http" to ensure it's configured correctly
    starlette_app = mcp.http_app(transport="streamable-http")
    wrapped_app = SessionCleanupMiddleware(starlette_app, registry)
    
    import uvicorn
    uvicorn.run(wrapped_app, host="0.0.0.0", port=port)
