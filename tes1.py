# coding: utf-8 
import json

str1 = 'Sen. Charles E. Schumer called on federal safety officials yesterday to reopen their investigation into the fatal crash of a passenger jet in Belle Harbor , Queens 1 , because equipment failure , not pilot error , might have been the cause .'
print(str1.find('Queens 1'))

'''
with open('dataset/riedel_train_bag.json') as f:
    store = {}
    for idx, data in enumerate(f):
        data = json.loads(data)
        subId, objId = data['subId'], data['objId']
        key = subId+'_'+objId
        if not store.get(key):
            store[key] = 1
        else:
            store[key] = store[key] + 1
for k,v in store.items():
    if v > 1:
        print(k)
'''

rel2id = json.loads(open('./dataset/riedel_relation2id.json').read())
id2rel = dict([(v, k) for k, v in rel2id.items()])
print(rel2id.get('/base/locations/countries/states_provinces_within',rel2id.get('NA')))