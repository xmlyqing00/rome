

packages = []
with open('requirements.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    packages.append(line.split('==')[0])

print(packages)

with open('requirements2.txt', 'w') as f:
    for p in packages:
        f.write(f'{p}\n')