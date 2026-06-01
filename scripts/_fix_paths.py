import os
import re

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)

py_files = sorted([
    f for f in os.listdir(SCRIPTS_DIR)
    if f.endswith('.py') and not f.startswith('_')
])

PATTERNS = [
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\([^'\"\\]+)['\"]",
     r'os.path.join(PROJECT_ROOT, "data", r"\1")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data['\"]",
     r'os.path.join(PROJECT_ROOT, "data")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\figures['\"]",
     r'os.path.join(PROJECT_ROOT, "figures")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation['\"]",
     r'PROJECT_ROOT'),
    (r"r['\"]E:\\Motion capture data of six jump-landing, fatigued and non-fatigued, after anterior cruciate ligament injury['\"]",
     r'os.environ.get("C3D_DATA_DIR", os.path.join(PROJECT_ROOT, "..", "c3d_data"))'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\features_raw\.csv['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "features_raw.csv")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\features_with_knee\.csv['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "features_with_knee.csv")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\features_combined\.csv['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "features_combined.csv")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\results_summary\.json['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "results_summary.json")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\acl_results_summary\.json['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "acl_results_summary.json")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\final_results\.md['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "final_results.md")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\final_results\.txt['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "final_results.txt")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\supplement_results\.json['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "supplement_results.json")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\supplement_results\.txt['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "supplement_results.txt")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\spec_curve_data\.csv['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "spec_curve_data.csv")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\spec_curve_joint\.json['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "spec_curve_joint.json")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\data\\features_acl_raw\.csv['\"]",
     r'os.path.join(PROJECT_ROOT, "data", "features_acl_raw.csv")'),
    (r"'d:/TC-Sport/motion_computation/data/'",
     r'os.path.join(PROJECT_ROOT, "data", "")'),
    (r"r['\"]d:\\TC-Sport\\motion_computation\\figures\\main\.pdf['\"]",
     r'os.path.join(PROJECT_ROOT, "figures", "main.pdf")'),
]

fixed_count = 0
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
            print(f"  SKIP {fname}: encoding error")
            continue
    except Exception as e:
        print(f"  SKIP {fname}: {e}")
        continue
    original = content

    for pat, repl in PATTERNS:
        content = re.sub(pat, repl, content)

    if 'PROJECT_ROOT' in content and 'PROJECT_ROOT =' not in content:
        lines = content.split('\n')
        last_import = 0
        for i, line in enumerate(lines):
            if re.match(r'^(import |from )', line):
                last_import = i + 1
        proj_root_line = f'PROJECT_ROOT = {repr(PROJECT_DIR)}'
        lines.insert(last_import, '')
        lines.insert(last_import, proj_root_line)
        if 'import os' not in original:
            lines.insert(last_import, 'import os')
        content = '\n'.join(lines)

    if content != original:
        with open(fpath, 'w', encoding='utf-8', newline='\n') as fo:
            fo.write(content)
        print(f"  FIXED: {fname}")
        fixed_count += 1

print(f"\nFixed {fixed_count} files.")
