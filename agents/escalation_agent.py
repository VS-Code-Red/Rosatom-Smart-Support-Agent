class EscalationAgent:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def decide(self, confidence):
        return confidence < self.threshold