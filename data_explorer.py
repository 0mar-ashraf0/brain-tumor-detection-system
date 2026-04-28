import os

base = 'dataset'
if os.path.exists(base):
    print('dataset contents:', os.listdir(base))
    for item in os.listdir(base):
        full = os.path.join(base, item)
        print(f'{item}: is_dir={os.path.isdir(full)}')
        if os.path.isdir(full):
            try:
                contents = os.listdir(full)
                print(f'  contents (first 5): {contents[:5]}')
                training_path = os.path.join(full, 'Training')
                if os.path.exists(training_path):
                    print(f'  Training found, classes: {os.listdir(training_path)}')
                classes_path = os.path.join(full, 'glioma') # test class
                if os.path.exists(classes_path):
                    print(f'  Direct classes found, sample files: {os.listdir(classes_path)[:3]}')
            except PermissionError:
                print('  Permission denied')
        print('---')
else:
    print('No dataset dir')

print('\\nCheck brain_tumor_data:')
if os.path.exists('brain_tumor_data'):
    print(os.listdir('brain_tumor_data'))

