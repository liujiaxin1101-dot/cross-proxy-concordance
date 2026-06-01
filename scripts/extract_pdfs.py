import pdfplumber, os, glob, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_dir = os.path.join(PROJECT_ROOT, "literature")
for pdf_path in sorted(glob.glob(os.path.join(pdf_dir, '*.pdf'))):
    fname = os.path.basename(pdf_path)
    print(f'\n{"="*80}')
    print(f'FILE: {fname}')
    print('='*80)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = min(len(pdf.pages), 25)
            for i in range(pages):
                text = pdf.pages[i].extract_text()
                if text:
                    print(f'\n--- Page {i+1} ---')
                    # Clean text
                    clean = text.encode('ascii', errors='replace').decode('ascii')
                    print(clean[:3000])
    except Exception as e:
        print(f'ERROR: {e}')
