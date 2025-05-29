import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger('mediapipe').setLevel(logging.ERROR)  

from eye_test_cv.config.logging_config import configure_logging
from eye_test_cv.controller import PostureDistanceDetector
from eye_test_cv.config.settings import KNOWN_FACE_WIDTH

def get_application_settings():
    print("\nApplication Settings")
    print("===================")
    
    # Logging settings
    print("\nLogging Options:")
    print("1. Detailed logging")
    print("2. No logging")
    
    while True:
        try:
            log_choice = input("Choose logging option (1-2): ").strip()
            if log_choice not in ['1', '2']:
                print("Invalid choice. Please enter 1-2.")
                continue
            break
        except Exception:
            print("Invalid input. Please try again.")
    
    # Performance metrics settings
    print("\nPerformance Metrics:")
    print("1. Detailed logging")
    print("2. No logging")
    
    while True:
        try:
            metrics_choice = input("Choose metrics option (1-2): ").strip()
            if metrics_choice not in ['1', '2']:
                print("Invalid choice. Please enter 1-2.")
                continue
            break
        except Exception:
            print("Invalid input. Please try again.")
    
    return log_choice == '1', metrics_choice == '1'

def get_calibration_settings():
    print("\nDistance Estimation Setup")
    print("=======================")
    print("Choose calibration method:")
    print("1. Automatic calibration")
    print("2. Manual settings")
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice not in ['1', '2']:
                print("Invalid choice. Please enter 1 or 2.")
                continue
            break
        except Exception:
            print("Invalid input. Please try again.")
    
    auto_calibrate = (choice == '1')
    gender = 'average'
    face_width = None
    
    if auto_calibrate:
        print("\nAutomatic calibration selected.")
        print("\nSelect gender for statistical face width estimation:")
        print("1. Male")
        print("2. Female")
        print("3. Average")
        
        while True:
            try:
                gender_choice = input("Enter your choice (1-3): ").strip()
                if gender_choice not in ['1', '2', '3']:
                    print("Invalid choice. Please enter 1, 2, or 3.")
                    continue
                break
            except Exception:
                print("Invalid input. Please try again.")
        
        gender_map = {'1': 'male', '2': 'female', '3': 'average'}
        gender = gender_map[gender_choice]
    else:
        print("\nManual calibration selected.")
        print(f"\nDefault face width is set to {KNOWN_FACE_WIDTH} cm.")
        while True:
            try:
                use_default = input("Use default face width? (y/n): ").strip().lower()
                if use_default not in ['y', 'n']:
                    print("Invalid input. Please enter 'y' or 'n'.")
                    continue
                break
            except Exception:
                print("Invalid input. Please try again.")
        
        if use_default == 'n':
            while True:
                try:
                    face_width = float(input("Enter custom face width in cm: "))
                    if face_width <= 0:
                        print("Face width must be positive.")
                        continue
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except Exception:
                    print("Invalid input. Please try again.")
    
    return auto_calibrate, gender, face_width

def main():
    # Get application settings first
    enable_logging, enable_metrics = get_application_settings()
    
    # Enable logging and metrics if selected
    if enable_logging:
        PostureDistanceDetector.enableLogging()
    if enable_metrics:
        PostureDistanceDetector.enableMetrics()
    
    # Get calibration settings
    auto_calibrate, gender, face_width = get_calibration_settings()
    
    # Initialize and run the application
    app = PostureDistanceDetector(
        auto_calibrate=auto_calibrate,
        gender=gender,
        face_width=face_width
    )
    app.run()

if __name__ == "__main__":
    main()