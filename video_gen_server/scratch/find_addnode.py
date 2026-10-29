import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\qvrex\AppData\Local\ai.griptape.nodes.desktop\current\resources\editor-bundle\assets\index-DEZU41rc.js', 'r', encoding='utf-8') as f:
    text = f.read()

pos = text.find('new _ae(')
print(text[pos-800:pos+200])
