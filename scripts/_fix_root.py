import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

py_files = [
    f for f in os.listdir(SCRIPTS_DIR)
    if f.endswith('.py') and not f.startswith('_')
]

old_double_bs = "PROJECT_ROOT = 'C:\\\\Users\\\\Chuy\\\\Desktop\\\\motion_computation'"
new_val = "PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"

count = 0
for fname in py_files:
    fpath = os.path.join(SCRIPTS_DIR, fname)
    try:
        with open(fpath, 'r', encoding='utf-8') as fo:
            content = fo.read()
    except UnicodeDecodeError:
        try:
            with open(fpath, 'r', encoding='gbk') as fo:
                content = fo.read()
        except UnicodeDecodeError:
            print(f"  SKIP {fname}")
            continue

    if old_double_bs not in content:
        continue

    content = content.replace(old_double_bs, new_val)
    with open(fpath, 'w', encoding='utf-8', newline='\n') as fo:
        fo.write(content)
    print(f"  FIXED: {fname}")
    count += 1

print(f"\nFixed {count} files.")
