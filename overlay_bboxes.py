import json
import fitz  # PyMuPDF

PDF_PATH = "mistral7b.pdf"
OUTPUT_PATH = "mistral7b_bboxes.pdf"
DATA_PATH = "output.json"

with open(DATA_PATH) as f:
    data = json.load(f)

doc = fitz.open(PDF_PATH)

for page_data in data["pages"]:
    page_idx = page_data["index"]
    if page_idx >= len(doc):
        continue
    page = doc[page_idx]
    pdf_w = page.rect.width   # in points (72 dpi)
    pdf_h = page.rect.height

    # Get OCR page dimensions (pixels) to compute scale factors
    dims = page_data.get("dimensions")
    if dims:
        scale_x = pdf_w / dims["width"]
        scale_y = pdf_h / dims["height"]
    else:
        scale_x = 1.0
        scale_y = 1.0

    for img in page_data["images"]:
        bb = img["bounding_box"]
        if any(v is None for v in bb.values()):
            continue

        # Scale OCR pixel coords to PDF point coords
        rect = fitz.Rect(
            bb["top_left_x"] * scale_x,
            bb["top_left_y"] * scale_y,
            bb["bottom_right_x"] * scale_x,
            bb["bottom_right_y"] * scale_y,
        )

        # Draw rectangle
        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=(1, 0, 0), width=2, fill=None)
        shape.commit()

        # Add label
        label = img["id"]
        if "annotation" in img:
            cat = img["annotation"].get("category", "")
            label = f"{img['id']} [{cat}]"

        text_point = fitz.Point(rect.x0, rect.y0 - 4)
        page.insert_text(text_point, label, fontsize=8, color=(1, 0, 0))

doc.save(OUTPUT_PATH)
doc.close()
print(f"Saved annotated PDF to {OUTPUT_PATH}")
