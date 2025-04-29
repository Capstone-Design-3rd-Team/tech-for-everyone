import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
import pytesseract
from PIL import ImageFont, ImageDraw

# Constants
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
MIN_CONFIDENCE = 60
PADDING = 20
CONFIDENCE_THRESHOLD = 0.6
LARGE_TEXT_SIZE = 24
MIN_TEXT_SIZE = 12
OCR_MIN_CONFIDENCE = 80
WINDOW_SIZE = 10

# WCAG Contrast Levels
WCAG_AAA = 7.0
WCAG_AA = 4.5
WCAG_A = 3.0

# Output Directories
OUTPUT_BASE_DIR = "detected_elements"
OUTPUT_DIRS = {
    "text": os.path.join(OUTPUT_BASE_DIR, "text_elements"),
    "other": os.path.join(OUTPUT_BASE_DIR, "other_elements"),
    "text_analysis": os.path.join(OUTPUT_BASE_DIR, "text_analysis"),
    "contrast_analysis": os.path.join(OUTPUT_BASE_DIR, "contrast_analysis")
}

# Model Paths
MODEL_PATHS = {
    "web": "webui/downloads/checkpoints/screenrecognition-web350k.torchscript",
    "vins": "webui/downloads/checkpoints/screenrecognition-web350k-vins.torchscript"
}

@dataclass
class ContrastResult:
    contrast_ratio: float
    position: Tuple[int, int]
    window_size: int
    min_luminance: float
    max_luminance: float
    wcag_level: str

@dataclass
class TextResult:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    font_size: int

@dataclass
class UIEvaluation:
    accessibility_score: int
    usability_score: int
    issues: List[str]
    recommendations: List[str]

class UIAnalyzer:
    def __init__(self, ui_type: str):
        self.ui_type = ui_type
        self.class_mapping = self._load_class_mapping()
        self._setup_directories()
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    def _setup_directories(self) -> None:
        """Set up output directories"""
        for dir_path in OUTPUT_DIRS.values():
            os.makedirs(dir_path, exist_ok=True)

    def _load_class_mapping(self) -> Dict[int, str]:
        """Load class mapping based on UI type"""
        if self.ui_type == "web":
            return {
                0: "BACKGROUND", 1: "OTHER", 2: "StaticText", 3: "link",
                4: "listitem", 5: "paragraph", 6: "heading", 7: "img",
                8: "LineBreak", 9: "generic", 10: "gridcell", 11: "button",
                12: "separator", 13: "time", 14: "LayoutTableCell",
                15: "LabelText", 16: "figure", 17: "textbox", 18: "list",
                19: "Iframe", 20: "Pre", 21: "strong", 22: "columnheader",
                23: "Canvas", 24: "DescriptionListTerm",
                25: "DescriptionListDetail", 26: "HeaderAsNonLandmark",
                27: "superscript", 28: "row", 29: "checkbox", 30: "Abbr",
                31: "code"
            }
        return {
            0: "BACKGROUND", 1: "OTHER", 2: "Background Image",
            3: "Checked View", 4: "Icon", 5: "Input Field", 6: "Image",
            7: "Text", 8: "Text Button", 9: "Page Indicator",
            10: "Pop-Up Window", 11: "Sliding Menu", 12: "Switch"
        }

    @staticmethod
    def calculate_relative_luminance(r: float, g: float, b: float) -> float:
        """Calculate relative luminance according to WCAG 2.0"""
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        
        def to_linear(c: float) -> float:
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        
        r, g, b = map(to_linear, [r, g, b])
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def calculate_contrast(self, image: np.ndarray) -> ContrastResult:
        """Calculate contrast ratio according to WCAG 2.0"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = image.shape[:2]
        max_contrast = 0
        max_contrast_pos = (0, 0)
        max_contrast_ratio = 1.0
        min_luminance = 0
        max_luminance = 0

        for y in range(0, h - WINDOW_SIZE, WINDOW_SIZE//2):
            for x in range(0, w - WINDOW_SIZE, WINDOW_SIZE//2):
                window = image[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                if window.size > 0:
                    luminances = [
                        self.calculate_relative_luminance(pixel[0], pixel[1], pixel[2])
                        for pixel in window.reshape(-1, 3)
                    ]
                    
                    curr_min = min(luminances)
                    curr_max = max(luminances)
                    
                    if curr_min > 0:
                        curr_contrast = (curr_max + 0.05) / (curr_min + 0.05)
                        if curr_contrast > max_contrast:
                            max_contrast = curr_contrast
                            max_contrast_pos = (x, y)
                            min_luminance = curr_min
                            max_luminance = curr_max
                            max_contrast_ratio = curr_contrast

        wcag_level = 'AAA' if max_contrast_ratio >= WCAG_AAA else 'AA' if max_contrast_ratio >= WCAG_AA else 'Fail'
        
        return ContrastResult(
            contrast_ratio=max_contrast_ratio,
            position=max_contrast_pos,
            window_size=WINDOW_SIZE,
            min_luminance=min_luminance,
            max_luminance=max_luminance,
            wcag_level=wcag_level
        )

    @staticmethod
    def load_and_preprocess_image(image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """Load and preprocess image"""
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        transform = torchvision.transforms.ToTensor()
        return transform(image), image

    def extract_text_and_font_size(self, image: Image.Image) -> List[TextResult]:
        """Extract text and font size from image using OCR"""
        if isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = image

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        ocr_data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT, lang='kor+eng')
        
        text_results = []
        for i in range(len(ocr_data['text'])):
            if not ocr_data['text'][i].strip() or int(ocr_data['conf'][i]) < MIN_CONFIDENCE:
                continue
                
            text_results.append(TextResult(
                text=ocr_data['text'][i],
                confidence=float(ocr_data['conf'][i]),
                x=ocr_data['left'][i],
                y=ocr_data['top'][i],
                width=ocr_data['width'][i],
                height=ocr_data['height'][i],
                font_size=ocr_data['height'][i]
            ))
        
        return text_results

    def evaluate_ui_element(self, class_name: str, contrast_result: ContrastResult,
                          text_results: Optional[List[TextResult]] = None) -> UIEvaluation:
        """Evaluate UI element accessibility and usability"""
        evaluation = UIEvaluation(
            accessibility_score=0,
            usability_score=0,
            issues=[],
            recommendations=[]
        )
        
        if text_results:
            for result in text_results:
                is_large_text = result.font_size >= LARGE_TEXT_SIZE
                
                if is_large_text:
                    if contrast_result.contrast_ratio >= WCAG_AA:
                        evaluation.accessibility_score += 3
                    elif contrast_result.contrast_ratio >= WCAG_A:
                        evaluation.accessibility_score += 2
                    else:
                        evaluation.issues.append(
                            f"큰 텍스트({result.font_size}px)의 명암 대비가 WCAG AA 기준을 만족하지 않습니다 "
                            f"(현재: {contrast_result.contrast_ratio:.2f}:1)"
                        )
                        evaluation.recommendations.append(
                            "큰 텍스트의 명암 대비를 3:1 이상으로 개선하세요"
                        )
                else:
                    if contrast_result.contrast_ratio >= WCAG_AAA:
                        evaluation.accessibility_score += 3
                    elif contrast_result.contrast_ratio >= WCAG_AA:
                        evaluation.accessibility_score += 2
                    else:
                        evaluation.issues.append(
                            f"일반 텍스트({result.font_size}px)의 명암 대비가 WCAG AA 기준을 만족하지 않습니다 "
                            f"(현재: {contrast_result.contrast_ratio:.2f}:1)"
                        )
                        evaluation.recommendations.append(
                            "일반 텍스트의 명암 대비를 4.5:1 이상으로 개선하세요"
                        )
                
                if result.font_size < MIN_TEXT_SIZE:
                    evaluation.issues.append(f"글자 크기가 작습니다 (현재: {result.font_size}px)")
                    evaluation.recommendations.append(f"글자 크기를 {MIN_TEXT_SIZE}px 이상으로 설정하세요")
                else:
                    evaluation.usability_score += 1
                
                if result.confidence < OCR_MIN_CONFIDENCE:
                    evaluation.issues.append(f"텍스트 인식 신뢰도가 낮습니다 (현재: {result.confidence}%)")
        else:
            if contrast_result.contrast_ratio >= WCAG_A:
                evaluation.accessibility_score += 2
            else:
                evaluation.issues.append(
                    f"UI 요소의 명암 대비가 WCAG 기준을 만족하지 않습니다 "
                    f"(현재: {contrast_result.contrast_ratio:.2f}:1)"
                )
                evaluation.recommendations.append(
                    "UI 요소의 명암 대비를 3:1 이상으로 개선하세요"
                )
        
        return evaluation

    def save_analysis_results(self, class_name: str, box: np.ndarray, score: float,
                            contrast_result: ContrastResult, text_results: Optional[List[TextResult]] = None,
                            evaluation: Optional[UIEvaluation] = None, index: int = 0) -> None:
        """Save analysis results to files"""
        if text_results and len(text_results) > 0:
            analysis_path = os.path.join(
                OUTPUT_DIRS["text_analysis"],
                f"{class_name}_{index}_{score:.2f}_analysis.json"
            )
            
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'class': class_name,
                    'confidence': float(score),
                    'position': [float(x) for x in box.tolist()],
                    'text_analysis': [vars(result) for result in text_results],
                    'contrast_analysis': vars(contrast_result),
                    'evaluation': vars(evaluation) if evaluation else None
                }, f, ensure_ascii=False, indent=2)
        else:
            contrast_path = os.path.join(
                OUTPUT_DIRS["contrast_analysis"],
                f"{class_name}_{index}_{score:.2f}_contrast.json"
            )
            
            with open(contrast_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'class': class_name,
                    'confidence': float(score),
                    'position': [float(x) for x in box.tolist()],
                    'contrast_analysis': vars(contrast_result)
                }, f, ensure_ascii=False, indent=2)

    def save_element_image(self, class_name: str, image: Image.Image,
                         score: float, index: int, is_text: bool = False) -> None:
        """Save element image to appropriate directory"""
        filename = f"{class_name}_{index}_{score:.2f}.png"
        output_dir = OUTPUT_DIRS["text"] if is_text else OUTPUT_DIRS["other"]
        save_path = os.path.join(output_dir, filename)
        image.save(save_path)

    def process_detected_element(self, element_data: Dict[str, Any], original_image: np.ndarray,
                               index: int) -> None:
        """Process a single detected UI element"""
        box = element_data['box']
        score = element_data['score']
        label = element_data['label']
        class_name = self.class_mapping.get(label, f"unknown-{label}")

        # Crop image with padding
        x1 = max(0, int(box[0]) - PADDING)
        y1 = max(0, int(box[1]) - PADDING)
        x2 = min(original_image.shape[1], int(box[2]) + PADDING)
        y2 = min(original_image.shape[0], int(box[3]) + PADDING)
        
        cropped_image = original_image[y1:y2, x1:x2]
        cropped_pil = Image.fromarray(cropped_image)
        
        # Analyze contrast
        contrast_result = self.calculate_contrast(cropped_image)
        
        # Check if text-related
        text_related = any(keyword in class_name.lower() for keyword in
                         ['text', 'label', 'title', 'heading', 'paragraph', 'image'])
        
        if text_related:
            text_results = self.extract_text_and_font_size(cropped_pil)
            if text_results and len(text_results) > 0:
                # Process text element
                evaluation = self.evaluate_ui_element(class_name, contrast_result, text_results)
                self.save_analysis_results(class_name, box, score, contrast_result,
                                        text_results, evaluation, index)
                self.save_element_image(class_name, cropped_pil, score, index, True)
                
                # Print evaluation results
                if evaluation.issues:
                    print("  발견된 문제점:")
                    for issue in evaluation.issues:
                        print(f"    - {issue}")
                if evaluation.recommendations:
                    print("  개선 권장사항:")
                    for rec in evaluation.recommendations:
                        print(f"    - {rec}")
            else:
                # Process as other element
                self.save_analysis_results(class_name, box, score, contrast_result)
                self.save_element_image(class_name, cropped_pil, score, index, False)
                print(f"  WCAG 명암 대비: {contrast_result.contrast_ratio:.2f}:1 "
                      f"(Level: {contrast_result.wcag_level})")
        else:
            # Process non-text element
            self.save_analysis_results(class_name, box, score, contrast_result)
            self.save_element_image(class_name, cropped_pil, score, index, False)
            print(f"  WCAG 명암 대비: {contrast_result.contrast_ratio:.2f}:1 "
                  f"(Level: {contrast_result.wcag_level})")

    def detect_ui_elements(self, image_path: str) -> None:
        """Detect and analyze UI elements in the image"""
        # Load model
        model = torch.jit.load(MODEL_PATHS[self.ui_type], map_location=torch.device('cpu'))
        model.eval()

        # Load and preprocess image
        image_tensor, original_image = self.load_and_preprocess_image(image_path)
        original_image_np = np.array(original_image)

        # Detect objects
        with torch.no_grad():
            losses, detections = model([image_tensor])
            
            # Visualization setup
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(original_image)
            
            print(f"\nUI 타입: {self.ui_type.upper()}")
            print(f"감지된 객체 수: {len(detections[0]['boxes'])}")
            print(f"상위 50개 감지 결과:")
            
            # Process each detection
            for i in range(min(50, len(detections[0]['boxes']))):
                if detections[0]['scores'][i] > CONFIDENCE_THRESHOLD:
                    element_data = {
                        'box': detections[0]['boxes'][i].numpy(),
                        'score': detections[0]['scores'][i].item(),
                        'label': detections[0]['labels'][i].item()
                    }
                    
                    # Draw bounding box
                    rect = patches.Rectangle(
                        (element_data['box'][0], element_data['box'][1]),
                        element_data['box'][2] - element_data['box'][0],
                        element_data['box'][3] - element_data['box'][1],
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    class_name = self.class_mapping.get(element_data['label'],
                                                      f"unknown-{element_data['label']}")
                    ax.text(
                        element_data['box'][0],
                        element_data['box'][1] - 5,
                        f"{class_name}: {element_data['score']:.2f}",
                        color='white',
                        fontsize=12,
                        bbox=dict(facecolor='red', alpha=0.5)
                    )
                    
                    print(f"  객체 {i+1}: {class_name}, 신뢰도 {element_data['score']:.2f}, "
                          f"위치 {element_data['box']}")
                    
                    # Process the detected element
                    self.process_detected_element(element_data, original_image_np, i)
            
            plt.axis('off')
            plt.title(f"UI Element Detection - {self.ui_type.upper()}")
            plt.show()

def main():
    image_path = "webui/example3.jpg"
    ui_type = input("UI 타입을 선택하세요 (web/vins): ").lower()
    
    if ui_type not in ["web", "vins"]:
        print("잘못된 선택입니다. 'web' 또는 'vins'를 입력하세요.")
        return
    
    analyzer = UIAnalyzer(ui_type)
    analyzer.detect_ui_elements(image_path)

if __name__ == "__main__":
    main() 
