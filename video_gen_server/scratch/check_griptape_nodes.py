import asyncio
import json
import websockets

async def check_registered_libraries():
    uri = "ws://127.0.0.1:18125"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as ws:
            req = {
                "request_id": "test_req_1",
                "request_type": "GetAllInfoForAllLibrariesRequest",
                "payload": {}
            }
            await ws.send(json.dumps(req))
            resp_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(resp_raw)
            print("Received response from engine!")
            lib_info = data.get("payload", {}).get("library_name_to_library_info", {})
            print(f"Total libraries registered: {len(lib_info)}")
            minimax_nodes = []
            for lib_name, lib_val in lib_info.items():
                node_defs = lib_val.get("nodes", [])
                for nd in node_defs:
                    name = nd.get("class_name")
                    disp = nd.get("metadata", {}).get("display_name")
                    desc = nd.get("metadata", {}).get("description")
                    if name == "MinimaxVideoGeneration" or (disp and "minimax" in disp.lower()):
                        minimax_nodes.append({
                            "library": lib_name,
                            "class_name": name,
                            "display_name": disp,
                            "desc": desc
                        })
            print(f"\n--- Found {len(minimax_nodes)} Minimax nodes registered in engine ---")
            for m in minimax_nodes:
                print(f"Library: '{m['library']}' | Class: {m['class_name']} | Display: {m['display_name']}")
    except Exception as e:
        print(f"Error querying websocket: {e}")

if __name__ == "__main__":
    asyncio.run(check_registered_libraries())
