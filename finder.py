from modulefinder import ModuleFinder

finder = ModuleFinder()
finder.run_script('main.py')

print('Loaded modules:')
# print(finder.modules)
for name, mod in finder.modules.items():
    if '.' not in name:
        print('%s: ' % name)
    # print(mod)
    # print(type(mod))
    # print(','.join(mod.globalnames[:3]))
