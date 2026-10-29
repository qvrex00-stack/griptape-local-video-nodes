with open(r'C:\Users\qvrex\AppData\Local\ai.griptape.nodes.desktop\current\resources\editor-bundle\assets\index-DEZU41rc.js', 'r', encoding='utf-8') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'`Tab`', text)]
print('Backtick Tab matches:', len(matches))
for pos in matches[:5]:
    print('--- Snippet ---')
    print(text[max(0, pos-100):min(len(text), pos+150)])
