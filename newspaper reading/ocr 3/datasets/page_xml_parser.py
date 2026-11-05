import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
from .base_loader import BaseOCRDatasetLoader, OCRItem

class PageXMLParser(BaseOCRDatasetLoader):
    """
    Parser for PAGE XML standard used by IMPACT Newspaper Ground Truth and PRImA datasets.
    Extracts text regions, text lines, and their exact unicode ground truth transcriptions.
    """

    def __init__(self, name: str = "PAGE XML Dataset"):
        super().__init__(name=name)

    def parse_points(self, points_str: str) -> Optional[Tuple[int, int, int, int]]:
        """Parses 'x1,y1 x2,y2 ...' into an enclosing bounding box (x, y, w, h)."""
        if not points_str:
            return None
        try:
            coords = []
            for pair in points_str.strip().split():
                if ',' in pair:
                    px, py = pair.split(',')
                    coords.append((int(float(px)), int(float(py))))
            if not coords:
                return None
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            return (min_x, min_y, max_x - min_x, max_y - min_y)
        except Exception:
            return None

    def parse_file(self, xml_path: str, image_dir: str) -> List[OCRItem]:
        items = []
        if not os.path.exists(xml_path):
            return items

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Handle XML namespaces dynamically
            ns = ""
            if "}" in root.tag:
                ns = root.tag.split("}")[0] + "}"

            page_elem = root.find(f".//{ns}Page")
            if page_elem is None:
                return items

            img_filename = page_elem.attrib.get("imageFilename", "")
            img_path = os.path.join(image_dir, img_filename)
            if not os.path.exists(img_path):
                # Try finding with xml basename
                base_name = os.path.splitext(os.path.basename(xml_path))[0]
                for ext in [".jpg", ".tif", ".tiff", ".png"]:
                    cand = os.path.join(image_dir, base_name + ext)
                    if os.path.exists(cand):
                        img_path = cand
                        break

            # Extract TextLine or TextRegion elements
            for region in root.findall(f".//{ns}TextRegion"):
                region_type = region.attrib.get("type", "paragraph")
                
                # Check for individual lines
                lines = region.findall(f".//{ns}TextLine")
                if lines:
                    for line in lines:
                        unicode_elem = line.find(f".//{ns}TextEquiv/{ns}Unicode")
                        if unicode_elem is not None and unicode_elem.text:
                            text = unicode_elem.text.strip()
                            coords_elem = line.find(f"{ns}Coords")
                            pts = coords_elem.attrib.get("points", "") if coords_elem is not None else ""
                            bbox = self.parse_points(pts)

                            items.append(OCRItem(
                                image_path=img_path,
                                transcription=text,
                                bbox=bbox,
                                dataset_source=self.name,
                                metadata={"type": "line", "region_type": region_type, "xml": xml_path}
                            ))
                else:
                    # Region-level ground truth
                    unicode_elem = region.find(f".//{ns}TextEquiv/{ns}Unicode")
                    if unicode_elem is not None and unicode_elem.text:
                        text = unicode_elem.text.strip()
                        coords_elem = region.find(f"{ns}Coords")
                        pts = coords_elem.attrib.get("points", "") if coords_elem is not None else ""
                        bbox = self.parse_points(pts)

                        items.append(OCRItem(
                            image_path=img_path,
                            transcription=text,
                            bbox=bbox,
                            dataset_source=self.name,
                            metadata={"type": "region", "region_type": region_type, "xml": xml_path}
                        ))
        except Exception as e:
            pass

        return items

    def load(self, data_dir: str) -> List[OCRItem]:
        """Loads all PAGE XML annotations in a directory."""
        items = []
        if not os.path.exists(data_dir):
            return items

        for root_p, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith('.xml'):
                    xml_path = os.path.join(root_p, f)
                    parsed = self.parse_file(xml_path, image_dir=root_p)
                    items.extend(parsed)

        return items
