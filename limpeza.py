import re
import json

with open('sumulas.json', 'r', encoding='utf-8') as f:
    sumulas_data = json.load(f)

for sumula in sumulas_data:
    sumula['texto'] = re.sub(r'\s*\([^)]+\)$', '', sumula['texto'])

with open('sumulas_limpo.json', 'w', encoding='utf-8') as f:
    json.dump(sumulas_data, f, ensure_ascii=False, indent=4)

print("Arquivo 'sumulas_limpo.json' salvo com sucesso!")
