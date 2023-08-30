#@title Insert paths to two settings.txt files to compare 

file1 = '55' #@param {'type':'string'}
file2 = '58' #@param {'type':'string'}

changes = []
added = []
removed = []

file1 = infer_settings_path(file1)
file2 = infer_settings_path(file2)

if file1 != '' and file2 != '':
  import json
  with open(file1, 'rb') as f:
    f1 = json.load(f)
  with open(file2, 'rb') as f:
    f2 = json.load(f)
  joint_keys = set(list(f1.keys())+list(f2.keys()))
  print(f'Comparing\n{file1.split("/")[-1]}\n{file2.split("/")[-1]}\n')
  for key in joint_keys:
    if key in f1.keys() and key in f2.keys() and f1[key] != f2[key]:
      changes.append(f'{key}: {f1[key]} -> {f2[key]}')
      # print(f'{key}: {f1[key]} -> {f2[key]}')
    if key in f1.keys() and key not in f2.keys():
      removed.append(f'{key}: {f1[key]} -> <variable missing>')
      # print(f'{key}: {f1[key]} -> <variable missing>')
    if key not in f1.keys() and key in f2.keys():
      added.append(f'{key}: <variable missing> -> {f2[key]}')
      # print(f'{key}: <variable missing> -> {f2[key]}')

print('Changed:\n')
for o in changes:
  print(o)

print('\n\nAdded in file2:\n')
for o in added:
  print(o)

print('\n\nRemoved in file2:\n')
for o in removed:
  print(o)