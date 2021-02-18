import logging

from helpers.files import create_folder

DEFAULT_PATH = '../logging'


# "Kind of a singleton/factory"
class Logger(object):
    """
    Returns loggers based on name, creates them if not existing
    """
    _instances = {}

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(self, name='coronawiki', path=DEFAULT_PATH):
        if name not in self._instances:
            filepath = f'{create_folder(path, "")}/{name}.log'
            print(f'New logging instance for {filepath}')
            logger = logging.getLogger(name)
            formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s', '%m-%d %H:%M')
            fileHandler = logging.FileHandler(filepath, mode='a')
            streamHandler = logging.StreamHandler()
            fileHandler.setFormatter(formatter)
            streamHandler.setFormatter(formatter)
            logger.addHandler(fileHandler)
            logger.addHandler(streamHandler)
            logger.setLevel(logging.DEBUG)
            logger.info('Created new singleton instance')
            self._instances[name] = logger

            # Put any initialization here.
        return self._instances[name]
