from code_1 import ExtremismDetector

detector = ExtremismDetector()

input_data = "I want this president."

extremism_score = detector.detect_extremism(input_data)
print(extremism_score)