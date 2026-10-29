import asyncio
import json
import uuid
import websockets

async def test_rpc():
    async with websockets.connect('ws://127.0.0.1:18125') as ws:
        req_id = str(uuid.uuid4())
        payload = {
            "event_type": "EventRequest",
            "request_type": "ListRegisteredLibrariesRequest",
            "request_id": req_id,
            "response_topic": "request",
            "request": {}
        }
        envelope = {
            "payload": payload,
            "type": "EventRequest",
            "topic": "request"
        }
        await ws.send(json.dumps(envelope))
        print("Sent ListRegisteredLibrariesRequest envelope")
        for _ in range(5):
            resp = await asyncio.wait_for(ws.recv(), timeout=3.0)
            data = json.loads(resp)
            print("Received:", json.dumps(data, indent=2)[:500])

if __name__ == '__main__':
    asyncio.run(test_rpc())
