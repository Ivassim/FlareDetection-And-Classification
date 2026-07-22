import pymupdf4llm
import glob
import os

pdf_dir = r"c:\Users\Rayan\Desktop\PFE\Memoire\anciens_memoires"
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

for pdf_path in pdf_files:
    print(f"Conversion de {os.path.basename(pdf_path)}...")
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        md_path = pdf_path.replace(".pdf", ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"-> Succès : {os.path.basename(md_path)}\n")
    except Exception as e:
        print(f"Erreur pour {os.path.basename(pdf_path)} : {str(e)}")

print("Termine.")
