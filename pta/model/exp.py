from abc import ABC, abstractmethod

class Experiment(ABC):
    def run(self):
        data = self.load_data()
        prepared = self.preprocess(data)
        result = self.pipeline(prepared)
        data = self.postprocess(result)
        return data

    @abstractmethod
    def pipeline(self, data):
        pass

    def load_data(self):
        return None

    def preprocess(self, data):
        return data

    def postprocess(self, result):
        return result