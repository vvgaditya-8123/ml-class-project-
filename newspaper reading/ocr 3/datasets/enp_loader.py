import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
from .base_loader import BaseOCRDatasetLoader, OCRItem

class ENPDatasetLoader(BaseOCRDatasetLoader):
    """
    Loader for the Electronic Newspaper Processing (ENP) Dataset (ICDAR 2015 format & ALTO XML).
    Parses complex multi-column newspaper layouts, text blocks, and line words with spatial coordinates.
    """

    def __init__(self):
        super().__init__(name="ENP Newspaper Dataset")

    def parse_alto_xml(self, xml_path: str, image_dir: str) -> List[OCRItem]:
        items = []
        if not os.path.exists(xml_path):
            return items

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            ns = ""
            if "}" in root.tag:
                ns = root.tag.split("}")[0] + "}"

            base_name = os.path.splitext(os.path.basename(xml_path))[0]
            img_path = os.path.join(image_dir, base_name + ".jpg")
            if not os.path.exists(img_path):
                for ext in [".png", ".tif", ".tiff", ".bmp"]:
                    cand = os.path.join(image_dir, base_name + ext)
                    if os.path.exists(cand):
                        img_path = cand
                        break

            # Find TextLine elements
            for line in root.findall(f".//{ns}TextLine"):
                hpos = int(float(line.attrib.get("HPOS", 0)))
                vpos = int(float(line.attrib.get("VPOS", 0)))
                width = int(float(line.attrib.get("WIDTH", 0)))
                height = int(float(line.attrib.get("HEIGHT", 0)))

                # Collect words (Strings)
                words = []
                for s in line.findall(f".//{ns}String"):
                    content = s.attrib.get("CONTENT", "")
                    if content:
                        words.append(content)

                if words:
                    line_text = " ".join(words).strip()
                    items.append(OCRItem(
                        image_path=img_path,
                        transcription=line_text,
                        bbox=(hpos, vpos, width, height),
                        dataset_source="ENP Dataset",
                        metadata={"type": "line", "format": "ALTO"}
                    ))
        except Exception:
            pass

        return items

    def load(self, data_dir: str) -> List[OCRItem]:
        items = []
        if not os.path.exists(data_dir):
            return items

        for root_p, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith('.xml'):
                    items.extend(self.parse_alto_xml(os.path.join(root_p, f), image_dir=root_p))

        return items
