import logging
import os

from fastmcp import Context, FastMCP


# Set up logging to a file
logging.basicConfig(
    filename="/home/vivekv/Documents/open-source-repos/software-agent-sdk/experiment/mcp_server/mcp_server.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

mcp = FastMCP("Experiment-Server")


@mcp.tool()
async def get_session_id(ctx: Context) -> str:
    """Returns the current session ID."""
    session_id = ctx.session_id
    logging.info(f"Tool called. Session ID: {session_id}")
    logging.info(f"Transport: {ctx.transport}")

    # Also log headers if available
    try:
        if ctx.request_context and ctx.request_context.request:
            request = ctx.request_context.request
            headers = dict(request.headers)
            logging.info(f"Headers: {headers}")
    except Exception as e:
        logging.info(f"Could not get headers: {e}")

    return session_id


if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", 8000))
    print(f"Starting Streamable HTTP server on port {port}")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
