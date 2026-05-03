import os

def count_images(folder):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return sum(1 for f in os.listdir(folder) if f.lower().endswith(extensions))

base = 'dataset'
if os.path.exists(base):
    print('dataset contents:', os.listdir(base))
    for item in os.listdir(base):
        full = os.path.join(base, item)
        print(f'\n{item}: is_dir={os.path.isdir(full)}')
        if os.path.isdir(full):
            try:
                contents = os.listdir(full)
                print(f'  contents (first 5): {contents[:5]}')

                # Check for Training folder
                training_path = os.path.join(full, 'Training')
                if os.path.exists(training_path):
                    print(f'  Training found:')
                    for cls in os.listdir(training_path):
                        cls_path = os.path.join(training_path, cls)
                        if os.path.isdir(cls_path):
                            count = count_images(cls_path)
                            print(f'    {cls}: {count} images')

                # Check for Train folder
                train_path = os.path.join(full, 'Train')
                if os.path.exists(train_path):
                    print(f'  Train found:')
                    for cls in os.listdir(train_path):
                        cls_path = os.path.join(train_path, cls)
                        if os.path.isdir(cls_path):
                            count = count_images(cls_path)
                            print(f'    {cls}: {count} images')

                # Check for Test folder
                test_path = os.path.join(full, 'Test')
                if os.path.exists(test_path):
                    print(f'  Test found:')
                    for cls in os.listdir(test_path):
                        cls_path = os.path.join(test_path, cls)
                        if os.path.isdir(cls_path):
                            count = count_images(cls_path)
                            print(f'    {cls}: {count} images')

                # Check for direct class folders
                classes_path = os.path.join(full, 'glioma')
                if os.path.exists(classes_path):
                    print(f'  Direct classes found:')
                    for cls in os.listdir(full):
                        cls_path = os.path.join(full, cls)
                        if os.path.isdir(cls_path):
                            count = count_images(cls_path)
                            print(f'    {cls}: {count} images')

            except PermissionError:
                print('  Permission denied')
        print('---')
else:
    print('No dataset dir')

print('\nCheck brain_tumor_data:')
if os.path.exists('brain_tumor_data'):
    for item in os.listdir('brain_tumor_data'):
        item_path = os.path.join('brain_tumor_data', item)
        if os.path.isdir(item_path):
            count = count_images(item_path)
            print(f'  {item}: {count} images')