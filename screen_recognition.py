import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
import pytesseract
from PIL import ImageFont, ImageDraw

# Tesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def calculate_relative_luminance(r, g, b):
    """
    WCAG 2.0 기준의 상대 휘도 계산
    """
    # RGB 값을 0-1 범위로 정규화
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    
    # sRGB에서 선형 RGB로 변환
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    
    # 상대 휘도 계산
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def calculate_contrast(image, window_size=10):
    """
    WCAG 2.0 기준의 명암 대비를 계산하는 함수
    window_size x window_size 크기의 윈도우를 사용하여 지역적 대비를 계산
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 윈도우 크기
    h, w = image.shape[:2]
    max_contrast = 0
    max_contrast_pos = (0, 0)
    max_contrast_ratio = 1.0
    min_luminance = 0
    max_luminance = 0
    
    # 윈도우를 이동하며 대비 계산
    for y in range(0, h - window_size, window_size//2):
        for x in range(0, w - window_size, window_size//2):
            window = image[y:y+window_size, x:x+window_size]
            if window.size > 0:
                # 윈도우 내의 모든 픽셀에 대해 상대 휘도 계산
                luminances = []
                for pixel in window.reshape(-1, 3):
                    luminance = calculate_relative_luminance(pixel[0], pixel[1], pixel[2])
                    luminances.append(luminance)
                
                # 최소/최대 휘도 찾기
                curr_min = min(luminances)
                curr_max = max(luminances)
                
                # WCAG 대비 계산
                if curr_min > 0:  # 0으로 나누기 방지
                    curr_contrast = (curr_max + 0.05) / (curr_min + 0.05)
                    
                    if curr_contrast > max_contrast:
                        max_contrast = curr_contrast
                        max_contrast_pos = (x, y)
                        min_luminance = curr_min
                        max_luminance = curr_max
                        max_contrast_ratio = curr_contrast
    
    return {
        'contrast_ratio': max_contrast_ratio,
        'position': max_contrast_pos,
        'window_size': window_size,
        'min_luminance': min_luminance,
        'max_luminance': max_luminance,
        'wcag_level': 'AAA' if max_contrast_ratio >= 7 else 'AA' if max_contrast_ratio >= 4.5 else 'Fail'
    }

# 클래스 매핑 로드
def load_class_mapping(mapping_type):
    if mapping_type == "web":
        # 웹 UI 클래스 매핑
        return {
            0: "BACKGROUND",
            1: "OTHER",
            2: "StaticText",
            3: "link",
            4: "listitem",
            5: "paragraph",
            6: "heading",
            7: "img",
            8: "LineBreak",
            9: "generic",
            10: "gridcell",
            11: "button",
            12: "separator",
            13: "time",
            14: "LayoutTableCell",
            15: "LabelText",
            16: "figure",
            17: "textbox",
            18: "list",
            19: "Iframe",
            20: "Pre",
            21: "strong",
            22: "columnheader",
            23: "Canvas",
            24: "DescriptionListTerm",
            25: "DescriptionListDetail",
            26: "HeaderAsNonLandmark",
            27: "superscript",
            28: "row",
            29: "checkbox",
            30: "Abbr",
            31: "code"
        }
    else:  # vins
        # VINS 데이터셋 클래스 매핑
        return {
            0: "BACKGROUND",
            1: "OTHER",
            2: "Background Image",
            3: "Checked View",
            4: "Icon",
            5: "Input Field",
            6: "Image",
            7: "Text",
            8: "Text Button",
            9: "Page Indicator",
            10: "Pop-Up Window",
            11: "Sliding Menu",
            12: "Switch"
        }

# 이미지 로드 및 전처리
def load_and_preprocess_image(image_path):
    # PIL로 이미지 로드
    image = Image.open(image_path)
    # RGB로 변환 (필요한 경우)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # 텐서로 변환 [C, H, W] 형식으로
    transform = torchvision.transforms.ToTensor()
    image_tensor = transform(image)
    return image_tensor, image

def extract_text_and_font_size(image, min_confidence=60):
    """
    이미지에서 텍스트와 글자 크기를 추출하는 함수
    """
    # OpenCV 이미지로 변환
    if isinstance(image, Image.Image):
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        cv_image = image
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # 이진화 (텍스트를 더 잘 인식하기 위해)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # OCR 실행 (한국어와 영어 모두 지원)
    ocr_data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT, lang='kor+eng')
    
    # 결과 저장
    text_results = []
    
    # 각 감지된 텍스트 블록에 대해
    for i in range(len(ocr_data['text'])):
        # 빈 텍스트 또는 낮은 신뢰도 무시
        if not ocr_data['text'][i].strip() or int(ocr_data['conf'][i]) < min_confidence:
            continue
        
        # 텍스트 정보 추출
        text = ocr_data['text'][i]
        conf = ocr_data['conf'][i]
        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]
        
        # 글자 크기 추정 (높이 기준)
        font_size = h
        
        # 결과 저장
        text_results.append({
            'text': text,
            'confidence': conf,
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'font_size': font_size
        })
    
    return text_results

def evaluate_ui_element(class_name, contrast_result, text_results=None):
    """
    UI 요소의 접근성과 사용성을 평가하는 함수
    """
    evaluation = {
        'accessibility_score': 0,
        'usability_score': 0,
        'issues': [],
        'recommendations': []
    }
    
    # 명암 대비 평가
    contrast_ratio = contrast_result['contrast_ratio']
    wcag_level = contrast_result['wcag_level']
    
    # 텍스트 요소 평가
    if text_results:
        for result in text_results:
            font_size = result['font_size']
            is_large_text = font_size >= 24  # 18pt/24px 기준
            
            # 텍스트 크기에 따른 WCAG 기준 적용
            if is_large_text:
                if contrast_ratio >= 4.5:
                    evaluation['accessibility_score'] += 3
                elif contrast_ratio >= 3:
                    evaluation['accessibility_score'] += 2
                else:
                    evaluation['issues'].append(
                        f"큰 텍스트({font_size}px)의 명암 대비가 WCAG AA 기준을 만족하지 않습니다 (현재: {contrast_ratio:.2f}:1)"
                    )
                    evaluation['recommendations'].append(
                        "큰 텍스트의 명암 대비를 3:1 이상으로 개선하세요"
                    )
            else:
                if contrast_ratio >= 7:
                    evaluation['accessibility_score'] += 3
                elif contrast_ratio >= 4.5:
                    evaluation['accessibility_score'] += 2
                else:
                    evaluation['issues'].append(
                        f"일반 텍스트({font_size}px)의 명암 대비가 WCAG AA 기준을 만족하지 않습니다 (현재: {contrast_ratio:.2f}:1)"
                    )
                    evaluation['recommendations'].append(
                        "일반 텍스트의 명암 대비를 4.5:1 이상으로 개선하세요"
                    )
            
            # 글자 크기 평가
            if font_size < 12:
                evaluation['issues'].append(f"글자 크기가 작습니다 (현재: {font_size}px)")
                evaluation['recommendations'].append("글자 크기를 12px 이상으로 설정하세요")
            else:
                evaluation['usability_score'] += 1
            
            # 텍스트 신뢰도 평가
            if result['confidence'] < 80:
                evaluation['issues'].append(f"텍스트 인식 신뢰도가 낮습니다 (현재: {result['confidence']}%)")
    else:
        # 텍스트가 아닌 UI 요소의 경우
        if contrast_ratio >= 3:
            evaluation['accessibility_score'] += 2
        else:
            evaluation['issues'].append(
                f"UI 요소의 명암 대비가 WCAG 기준을 만족하지 않습니다 (현재: {contrast_ratio:.2f}:1)"
            )
            evaluation['recommendations'].append(
                "UI 요소의 명암 대비를 3:1 이상으로 개선하세요"
            )
    
    return evaluation

def detect_ui_elements(image_path, ui_type="vins", confidence_threshold=0.6, padding=20):
    # UI 타입에 따른 모델 선택
    if ui_type == "web":
        model_path = "downloads/checkpoints/screenrecognition-web350k.torchscript"
    else:  # vins
        model_path = "downloads/checkpoints/screenrecognition-web350k-vins.torchscript"
    
    # 클래스 매핑 로드
    class_mapping = load_class_mapping(ui_type)
    
    # 모델 로드
    m = torch.jit.load(model_path, map_location=torch.device('cpu'))
    m.eval()

    # 이미지 로드 및 전처리
    image_tensor, original_image = load_and_preprocess_image(image_path)
    
    # 출력 디렉토리 생성
    base_output_dir = "detected_elements"
    text_output_dir = os.path.join(base_output_dir, "text_elements")
    other_output_dir = os.path.join(base_output_dir, "other_elements")
    os.makedirs(text_output_dir, exist_ok=True)
    os.makedirs(other_output_dir, exist_ok=True)
    
    # 텍스트 분석 결과 저장 디렉토리
    text_analysis_dir = os.path.join(base_output_dir, "text_analysis")
    os.makedirs(text_analysis_dir, exist_ok=True)
    
    # 대비 분석 결과 저장 디렉토리
    contrast_analysis_dir = os.path.join(base_output_dir, "contrast_analysis")
    os.makedirs(contrast_analysis_dir, exist_ok=True)

    # 모델 실행
    with torch.no_grad():
        losses, detections = m([image_tensor])
        
        # 결과 시각화
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(original_image)
        
        # 감지된 객체 표시
        boxes = detections[0]['boxes']
        scores = detections[0]['scores']
        labels = detections[0]['labels']
        
        print(f"\nUI 타입: {ui_type.upper()}")
        print(f"감지된 객체 수: {len(boxes)}")
        print(f"상위 50개 감지 결과:")
        
        # 원본 이미지를 numpy 배열로 변환
        original_image_np = np.array(original_image)
        
        # 바운딩 박스 그리기 및 이미지 저장
        for i in range(min(50, len(boxes))):
            if scores[i] > confidence_threshold:
                box = boxes[i].numpy()
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                label_num = labels[i].item()
                class_name = class_mapping.get(label_num, f"unknown-{label_num}")
                ax.text(
                    box[0], box[1] - 5, 
                    f"{class_name}: {scores[i]:.2f}",
                    color='white', fontsize=12, 
                    bbox=dict(facecolor='red', alpha=0.5)
                )
                print(f"  객체 {i+1}: {class_name}, 신뢰도 {scores[i]:.2f}, 위치 {boxes[i]}")
                
                # 패딩을 추가한 박스 좌표 계산
                x1 = max(0, int(box[0]) - padding)
                y1 = max(0, int(box[1]) - padding)
                x2 = min(original_image_np.shape[1], int(box[2]) + padding)
                y2 = min(original_image_np.shape[0], int(box[3]) + padding)
                
                # 이미지 자르기
                cropped_image = original_image_np[y1:y2, x1:x2]
                
                # PIL Image로 변환
                cropped_pil = Image.fromarray(cropped_image)
                
                # 명암 대비 분석
                contrast_result = calculate_contrast(cropped_image)
                
                # 텍스트 관련 요소인지 확인
                text_related = any(keyword in class_name.lower() for keyword in ['text', 'label', 'title', 'heading', 'paragraph'])
                
                if text_related:
                    # OCR로 텍스트와 글자 크기 추출
                    text_results = extract_text_and_font_size(cropped_pil)
                    
                    if text_results:
                        # 텍스트 분석 결과를 바탕으로 이미지 크롭 조정
                        min_x = min(result['x'] for result in text_results)
                        min_y = min(result['y'] for result in text_results)
                        max_x = max(result['x'] + result['width'] for result in text_results)
                        max_y = max(result['y'] + result['height'] for result in text_results)
                        
                        # 텍스트 영역에 패딩 추가
                        padding_x = padding
                        padding_y = padding
                        new_x1 = max(0, min_x - padding_x)
                        new_y1 = max(0, min_y - padding_y)
                        new_x2 = min(cropped_image.shape[1], max_x + padding_x)
                        new_y2 = min(cropped_image.shape[0], max_y + padding_y)
                        
                        # 텍스트 영역만 크롭
                        cropped_image = cropped_image[new_y1:new_y2, new_x1:new_x2]
                        cropped_pil = Image.fromarray(cropped_image)
                    
                    # 분석 결과 저장
                    analysis_filename = f"{class_name}_{i}_{scores[i]:.2f}_analysis.json"
                    analysis_path = os.path.join(text_analysis_dir, analysis_filename)
                    
                    # UI 요소 평가
                    evaluation = evaluate_ui_element(class_name, contrast_result, text_results)
                    
                    # 분석 결과에 평가 정보 추가
                    with open(analysis_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'class': class_name,
                            'confidence': float(scores[i]),
                            'position': [float(x) for x in box.tolist()],
                            'text_analysis': text_results,
                            'contrast_analysis': {
                                'contrast_ratio': float(contrast_result['contrast_ratio']),
                                'position': [int(x) for x in contrast_result['position']],
                                'window_size': int(contrast_result['window_size']),
                                'min_value': int(contrast_result['min_luminance']),
                                'max_value': int(contrast_result['max_luminance'])
                            },
                            'evaluation': evaluation
                        }, f, ensure_ascii=False, indent=2)
                
                    if evaluation['issues']:
                        print("  발견된 문제점:")
                        for issue in evaluation['issues']:
                            print(f"    - {issue}")
                    if evaluation['recommendations']:
                        print("  개선 권장사항:")
                        for rec in evaluation['recommendations']:
                            print(f"    - {rec}")
                    
                    # 텍스트 요소 저장
                    filename = f"{class_name}_{i}_{scores[i]:.2f}.png"
                    save_path = os.path.join(text_output_dir, filename)
                    cropped_pil.save(save_path)
                else:
                    # 대비 분석 결과 저장
                    contrast_filename = f"{class_name}_{i}_{scores[i]:.2f}_contrast.json"
                    contrast_path = os.path.join(contrast_analysis_dir, contrast_filename)
                    with open(contrast_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'class': class_name,
                            'confidence': float(scores[i]),
                            'position': [float(x) for x in box.tolist()],
                            'contrast_analysis': {
                                'contrast_ratio': float(contrast_result['contrast_ratio']),
                                'position': [int(x) for x in contrast_result['position']],
                                'window_size': int(contrast_result['window_size']),
                                'min_value': int(contrast_result['min_luminance']),
                                'max_value': int(contrast_result['max_luminance'])
                            }
                        }, f, ensure_ascii=False, indent=2)
                    
                    print(f"  WCAG 명암 대비: {contrast_result['contrast_ratio']:.2f}:1 (Level: {contrast_result['wcag_level']})")
                    
                    # 텍스트가 아닌 요소 저장
                    filename = f"{class_name}_{i}_{scores[i]:.2f}.png"
                    save_path = os.path.join(other_output_dir, filename)
                    cropped_pil.save(save_path)

        plt.axis('off')
        plt.title(f"UI Element Detection - {ui_type.upper()}")
        plt.show()

if __name__ == "__main__":
    # 이미지 경로 지정
    image_path = "example.jpg"
    
    # UI 타입 선택 ("web" 또는 "vins")
    ui_type = input("UI 타입을 선택하세요 (web/vins): ").lower()
    if ui_type not in ["web", "vins"]:
        print("잘못된 선택입니다. 'web' 또는 'vins'를 입력하세요.")
        exit()
    
    # UI 요소 감지 실행
    detect_ui_elements(image_path, ui_type=ui_type, confidence_threshold=0.6, padding=20) 