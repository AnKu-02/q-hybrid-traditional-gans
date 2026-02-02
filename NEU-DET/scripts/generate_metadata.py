from pathlib import Path
import pandas as pd

ROOT = Path(r"/Users/ananyakulkarni/Desktop/q hybrid traditional gans/NEU-DET")
classes = ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]
class_to_id = {c:i for i,c in enumerate(classes)}
g2 = {"crazing","patches","scratches"}
def group(c): return "G2" if c in g2 else "G1"

rows = []
for split in ["train", "validation"]:
    for c in classes:
        img_dir = ROOT / split / "images" / c
        for img_path in sorted(img_dir.glob("*.jpg")):
            xml_path = ROOT / split / "annotations" / (img_path.stem + ".xml")
            rows.append({
                "path": str(img_path),
                "split": split,
                "class_name": c,
                "class_id": class_to_id[c],
                "group": group(c),
                "xml_path": str(xml_path) if xml_path.exists() else ""
            })

df = pd.DataFrame(rows)
df.to_csv(ROOT / "neu_metadata.csv", index=False)
print(df.groupby(["split","class_name"]).size())
