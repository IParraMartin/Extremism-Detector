from code_1 import ExtremismDetector

detector = ExtremismDetector()

input_data = 'I want to kill the president. The war must begin!'

extremism_score = detector.detect_extremism(input_data)

print(extremism_score)