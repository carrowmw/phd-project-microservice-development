import asyncio
import json
import websockets


# Define an asynchronous function named wsListener
async def wsListener():
    # Define the URLs for WebSocket connection and Urban Observatory API
    url = "wss://api.newcastle.urbanobservatory.ac.uk/stream"
    uo_url = "https://file.newcastle.urbanobservatory.ac.uk/"

    # Connect to the WebSocket server
    async with websockets.connect(url) as websocket:
        # Start an infinite loop to continuously listen for messages from the WebSocket
        while True:
            # Asynchronously sleep for a short duration to allow other tasks to run
            asyncio.sleep(0.01)

            # Receive a message from the WebSocket
            msg = await websocket.recv()

            # Parse the received message as JSON
            msg = json.loads(msg)

            # Check if the message contains "data"
            if "data" in msg:
                data = msg["data"]

                # Check if "brokerage" is present in the data
                if "brokerage" in data:
                    brokerage = data["brokerage"]

                    # Check if "broker" is present in the brokerage
                    if "broker" in brokerage:
                        broker = brokerage["broker"]

                        # Check if the ID of the broker is "UTMC Open Camera Feeds"
                        if broker["id"] == "UTMC Open Camera Feeds":
                            # Split the ID of brokerage to get the location
                            location = brokerage["id"].split(":")[0]

                            print(location)

                            # Extract timestamp and URL from the received data
                            dt = data["timeseries"]["value"]["time"]
                            url = data["timeseries"]["value"]["data"]

                            # Replace part of the URL with the uo_url
                            url = url.replace("public", uo_url)

                            # Print timestamp, location, and modified URL
                            print(dt, location, url)


# Entry point of the script
if __name__ == "__main__":
    # Run the asynchronous function wsListener() until it completes
    asyncio.get_event_loop().run_until_complete(wsListener())
