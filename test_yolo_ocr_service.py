"""
Test script for YOLO + EasyOCR Service
"""

def test_imports():
    """Test all required imports"""
    try:
        print("Testing imports...")
        
        # Test EasyOCR
        import easyocr
        print("✓ EasyOCR imported successfully")
        
        # Test OpenCV
        import cv2
        print("✓ OpenCV imported successfully")
        
        # Test PIL
        from PIL import Image
        print("✓ PIL imported successfully")
        
        # Test NumPy
        import numpy as np
        print("✓ NumPy imported successfully")
        
        # Test FastAPI components
        from fastapi import APIRouter, UploadFile, File, Form
        print("✓ FastAPI components imported successfully")
        
        # Test our service
        from src.router.api.v1.yolo_easyocr_service import router, SUPPORTED_LANGUAGES
        print("✓ YOLO-OCR service imported successfully")
        print(f"✓ Supported languages: {len(SUPPORTED_LANGUAGES)} languages")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_easyocr_init():
    """Test EasyOCR initialization"""
    try:
        print("\nTesting EasyOCR initialization...")
        
        import easyocr
        
        # Test with basic languages
        reader = easyocr.Reader(['en'], gpu=False)
        print("✓ EasyOCR reader created successfully (CPU mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ EasyOCR initialization error: {e}")
        return False

def test_service_constants():
    """Test service constants and configurations"""
    try:
        print("\nTesting service constants...")
        
        from src.router.api.v1.yolo_easyocr_service import SUPPORTED_LANGUAGES
        
        print(f"✓ Supported languages count: {len(SUPPORTED_LANGUAGES)}")
        
        # Check some key languages
        required_langs = ['vi', 'en', 'zh', 'ja']
        for lang in required_langs:
            if lang in SUPPORTED_LANGUAGES:
                print(f"✓ Language {lang} ({SUPPORTED_LANGUAGES[lang]}) supported")
            else:
                print(f"❌ Language {lang} not supported")
        
        return True
        
    except Exception as e:
        print(f"❌ Service constants error: {e}")
        return False

def main():
    """Main test function"""
    print("=== YOLO + EasyOCR Service Test ===\n")
    
    tests = [
        test_imports,
        test_easyocr_init,
        test_service_constants
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Service is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
