class Singleton:
    __instance = None

    @classmethod
    def __get_instance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kwargs):
        cls.__instance = cls(*args, **kwargs)
        cls.instance = cls.__get_instance
        return cls.__instance


class EventManager(Singleton):
    def __init__(self):
        self.__eventQueue = []

    def get_event(self):
        if len(self.__eventQueue) == 0:
            return None
        else:
            return self.__eventQueue.pop(0)

    def peek_event(self):
        if len(self.__eventQueue) == 0:
            return None
        else:
            return self.__eventQueue[0]

    def add_event(self, event):
        return self.__eventQueue.append(event)

    def notify(self, event):
        if event.get_event_type() == "Fire":
            location = event.get_camera_location()
            print(location + " : Fire Detected!!!")
        elif event.get_event_type() == "Intrusion":
            location = event.get_camera_location()
            print(location + " : Intrusion Detected!!!")
        else:
            print("Unknown event")
