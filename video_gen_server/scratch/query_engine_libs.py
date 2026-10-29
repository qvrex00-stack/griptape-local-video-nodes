import asyncio
import json
import websockets
from griptape_nodes.retained_mode.events.library_events import GetAllInfoForAllLibrariesRequest

async def run():
    try:
        async with websockets.connect('ws://127.0.0.1:18125') as ws:
            req = GetAllInfoForAllLibrariesRequest()
            msg = req.model_dump_json()
            print('Sending request...')
            await ws.send(msg)
            for _ in range(10):
                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(resp)
                typ = data.get('type') or data.get('event_type') or data.get('payload_type') or type(data)
                print('Received message of type/key:', typ, list(data.keys())[:5])
                payload = data.get('payload', data)
                lib_dict = payload.get('library_name_to_library_info', {})
                if lib_dict:
                    print(f'Total libraries count: {len(lib_dict)}')
                    for name, info in lib_dict.items():
                        nodes = info.get('nodes', [])
                        mm = [n for n in nodes if 'Minimax' in n.get('class_name', '')]
                        if mm:
                            print(f'  -> Library "{name}" has {len(mm)} Minimax nodes:')
                            for m in mm:
                                print(f'       class={m.get("class_name")}, display_name={m.get("metadata", {}).get("display_name")}')
                    break
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(run())
