import fitz
import re
import os



def to_txt(number):
    path = f"pdfs/{number}.pdf"
    doc = None
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"Error opening {path}: {e}")
        return

    os.makedirs(f"txts/{number}", exist_ok=True)
    for page in doc:
        text = page.get_text()
        text = re.sub(r"\s+", "", text).strip()
        text = text.replace("。", "\n")
        with open(f"txts/{number}/{page.number+1}.txt", "w", encoding="utf-8") as f:
            f.write(text)


for i in range(212, 389):
    to_txt(i)