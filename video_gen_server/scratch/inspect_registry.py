import sys
sys.path.insert(0, r'C:\Users\qvrex\AppData\Local\ai.griptape.nodes.desktop\current\resources\engine-bundle\python\Lib\site-packages')

from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.managers.library_manager import LibraryManager

# Initialize LibraryManager and discover libraries
import asyncio
from griptape_nodes.files.path_utils import resolve_workspace_path
from griptape_nodes.retained_mode.managers.settings import Settings

async def test():
    lm = LibraryManager()
    libs = LibraryRegistry.list_libraries()
    print('Libraries in registry initially:', libs)

if __name__ == '__main__':
    asyncio.run(test())
