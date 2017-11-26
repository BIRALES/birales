class Event:
    def __init__(self, message):
        self.message = message

    def to_json(self):
        return self.message.to_json()
