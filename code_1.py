class ExtremismDetector:

    ingroup_dict = ['us', 'we', 'our', 'ours', 'ourselves', 'ourself', 'ourselves']
    violence_dict = ['attack', 'war', 'violent', 'violence', 'kill', 'killing', 'killed']
    authoritarianism_dict = ['leader', 'authority', 'control', 'power', 'rule', 'dominate', 'domination', 'dominant', 'dominates', 'dominating']
    emotionality_dict = ['hate', 'fear', 'anger', 'angry', 'fearful', 'fearfully', 'fearfulness', 'rigged']
    negations = ["don't", "didn't", "doesn't", "isn't", "wasn't", "weren't", "aren't", "couldn't", 
                 "wouldn't", "shouldn't", "won't", "can't", "mightn't", "mustn't", "needn't", "never", 
                 "no", "none", "not"]

    
    weights = [1.0, 1.5, 1.2, 1.0]


    def __init__(self):

        try:
            import numpy as np
            self.np = np
    
        except:
            print("Numpy is not installed. Try pip install numpy")


    def _calculate_features(self, input_data):

        ingroup_count = 0
        violence_count = 0
        authoritarianism_count = 0
        emotionality_count = 0
        counterbalance = 0

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

            if word in self.negations:
                
                counterbalance += 1

        feature_vector = self.np.array([ingroup_count, violence_count, authoritarianism_count, emotionality_count])

        return feature_vector, counterbalance


    def _apply_logistic_function(self, score):

        return 1 / (1 + self.np.exp(-score))


    def detect_extremism(self, input_data):

        features, counterbalance = self._calculate_features(input_data)
        weighted_features = self.np.dot(features, self.weights)
        adjusted_weighted_features = weighted_features - counterbalance
        extremism_score = self._apply_logistic_function(adjusted_weighted_features)

        return extremism_score