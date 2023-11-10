from progressbar import ProgressBar


class Progress_Bar:

    def __init__(self, max_value):
        self.__bar = ProgressBar(max_value=max_value)
        self.__current_value = 0

    def incrementCurrentValue(self):
        self.__current_value += 1

    def showBar(self):
        self.__bar.update(self.__current_value)
