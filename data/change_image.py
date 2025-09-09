import fitz
import re
import os

def change_image(number):
    path = f"pdfs/{number}.pdf"
    doc = None
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"Error opening {path}: {e}")
        return

    os.makedirs(f"images/{number}", exist_ok=True)
    for page in doc:
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = f"images/{number}/page{page.number+1}_img{img_index+1}.{image_ext}"
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

for i in range(212, 389):
    change_image(i)