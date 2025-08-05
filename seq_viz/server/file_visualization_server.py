import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import websockets

from ..core import TrainingDataReader


class FileVisualizationServer:
    """WebSocket server that reads training data from JSONL files for visualization."""

    def __init__(
        self,
        data_file: str = "training_data.jsonl",
        host: str = "localhost",
        port: int = 8765,
        update_interval: float = 2.0,
    ):
        """
        Initialize the visualization server.

        Args:
            data_file: Path to the JSONL file to read from
            host: WebSocket server host
            port: WebSocket server port
            update_interval: How often to check for new data (seconds)
        """
        self.data_file = Path(data_file)
        self.host = host
        self.port = port
        self.update_interval = update_interval
        self.clients = set()
        self.last_step_sent = -1

    async def register_client(self, websocket):
        """Register a new client connection."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")

    async def unregister_client(self, websocket):
        """Unregister a client connection."""
        self.clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")

    async def send_to_all_clients(self, data: Dict[str, Any]):
        """Send data to all connected clients."""
        if self.clients:
            message = json.dumps(data)
            # Send to all clients concurrently
            await asyncio.gather(
                *[client.send(message) for client in self.clients], return_exceptions=True
            )

    async def get_latest_unsent_step(self) -> Optional[Dict[str, Any]]:
        """Get the latest step that hasn't been sent yet."""
        if not self.data_file.exists():
            return None

        try:
            reader = TrainingDataReader(str(self.data_file))

            # Find steps newer than last sent
            latest_step = None
            for entry in reader.iter_steps():
                if entry.get("step", -1) > self.last_step_sent:
                    latest_step = entry

            return latest_step
        except Exception as e:
            print(f"Error reading data file: {e}")
            return None

    async def monitor_file(self):
        """Monitor the data file for new entries and send updates."""
        print(f"Monitoring {self.data_file} for updates...")
        print(
            "Open file:///Users/chris/Documents/FAR/Projects/f/seq-viz/seq_viz/web/enhanced_dashboard.html"
        )

        while True:
            try:
                # Check for new data
                latest_step = await self.get_latest_unsent_step()

                if latest_step:
                    step_num = latest_step.get("step", -1)
                    print(f"Sending step {step_num} to {len(self.clients)} clients")

                    # Add type field for compatibility with existing dashboard
                    latest_step["type"] = "training_update"

                    await self.send_to_all_clients(latest_step)
                    self.last_step_sent = step_num

                # Wait before checking again
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                print(f"Error in monitor loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def handle_client(self, websocket, path):
        """Handle a client WebSocket connection."""
        await self.register_client(websocket)

        try:
            # Send initial data if available
            if self.data_file.exists():
                reader = TrainingDataReader(str(self.data_file))
                latest = reader.get_latest()

                if latest:
                    latest["type"] = "training_update"
                    await websocket.send(json.dumps(latest))
                    self.last_step_sent = latest.get("step", -1)

            # Keep connection alive and handle any client messages
            async for message in websocket:
                # Handle client requests if needed
                try:
                    data = json.loads(message)
                    if data.get("type") == "get_step":
                        step_num = data.get("step")
                        reader = TrainingDataReader(str(self.data_file))
                        step_data = reader.get_step(step_num)
                        if step_data:
                            step_data["type"] = "training_update"
                            await websocket.send(json.dumps(step_data))
                except json.JSONDecodeError:
                    pass

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

    async def start(self):
        """Start the WebSocket server."""
        print(f"Starting visualization server on ws://{self.host}:{self.port}")

        # Start the file monitor task
        monitor_task = asyncio.create_task(self.monitor_file())

        # Start the WebSocket server
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"Server running. Connect to ws://{self.host}:{self.port}")
            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                print("\nShutting down server...")
                monitor_task.cancel()


def main():
    """Run the visualization server."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualization server for training data files")
    parser.add_argument("--file", default="training_data.jsonl", help="JSONL file to read from")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Update check interval (seconds)"
    )

    args = parser.parse_args()

    server = FileVisualizationServer(
        data_file=args.file, host=args.host, port=args.port, update_interval=args.interval
    )

    asyncio.run(server.start())


if __name__ == "__main__":
    main()
