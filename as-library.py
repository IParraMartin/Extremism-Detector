from code import ExtremismDetector

detector = ExtremismDetector()

input_data = "I don't want to kill the president."

extremism_score = detector.detect_extremism(input_data)
print(extremism_score)