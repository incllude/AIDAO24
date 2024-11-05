import json

summary = {}

for i in range(160):
    with open('data_merge/merge_{}.json'.format(i), 'r') as f:
        data = json.load(f)
        summary = summary | data
with open('data/bn_to_sch.json', 'w') as f:
    json.dump(summary, f)