import numpy as np

class ExtremismDetector:

    def __init__(self, ingroup_dict, violence_dict, authoritarianism_dict, emotionality_dict, weights):

        self.ingroup_dict = set(ingroup_dict)
        self.violence_dict = set(violence_dict)
        self.authoritarianism_dict = set(authoritarianism_dict)
        self.emotionality_dict = set(emotionality_dict)
        self.weights = np.array(weights)


    def _calculate_features(self, input_data):

        ingroup_count = 0
        violence_count = 0
        authoritarianism_count = 0
        emotionality_count = 0

        words = input_data.lower().split()

        for word in words:
            if word in self.ingroup_dict:
                ingroup_count += 1
            if word in self.violence_dict:
                violence_count += 1
            if word in self.authoritarianism_dict:
                authoritarianism_count += 1
            if word in self.emotionality_dict:
                emotionality_count += 1

        feature_vector = np.array([ingroup_count, violence_count, authoritarianism_count, emotionality_count])
        return feature_vector


    def _apply_logistic_function(self, score):
        return 1 / (1 + np.exp(-score))


    def detect_extremism(self, input_data):
        features = self._calculate_features(input_data)
        weighted_features = np.dot(features, self.weights)
        extremism_score = self._apply_logistic_function(weighted_features)
        return extremism_score


ingroup_dict = ["us", "our", "we"]
violence_dict = ["attack", "war", "fight"]
authoritarianism_dict = ["leader", "authority", "control"]
emotionality_dict = ["hate", "anger", "fear"]

weights = [1.0, 1.5, 1.2, 1.3]

detector = ExtremismDetector(ingroup_dict, violence_dict, authoritarianism_dict, emotionality_dict, weights)

input_data = "We must attack the enemy and show them the power of our leader. Our anger will fuel us in this war."

extremism_score = detector.detect_extremism(input_data)
print("Extremism score:", extremism_score)
