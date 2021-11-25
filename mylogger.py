import time


class logger(object):
    """
        class docs
    """

    def __init__(self, params):
        """
            Constructor
        """
        pass

    @classmethod
    def debug(cls, *messageinfo):
        print(time.strftime("%Y-%m-%d %H:%M:%S"), *messageinfo)

    @classmethod
    def info(cls, *messageinfo):
        print(time.strftime("%Y-%m-%d %H:%M:%S"), *messageinfo)

    @classmethod
    def error(cls, *messageinfo):
        print(time.strftime("%Y-%m-%d %H:%M:%S"), *messageinfo)
