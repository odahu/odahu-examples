import os.path
import tarfile
import tempfile
from glob import glob
from html.parser import HTMLParser

import logging
import re
import sys
from typing import List, Tuple
from urllib.request import urlretrieve

LOGGER = logging.getLogger(__name__)
DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                'reuters21578-mld/reuters21578.tar.gz')
ARCHIVE_FILENAME = 'reuters21578.tar.gz'


class ReutersParser(HTMLParser):
    """Utility class to parse a SGML file and yield documents one at a time."""

    def __init__(self, encoding='latin-1'):
        HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.body = re.sub(r'\s+', r' ', self.body)
        self.docs.append({'title': self.title,
                          'body': self.body,
                          'topics': self.topics})
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.in_topic_d = 0
        if self.in_topics:
            self.topics.append(self.topic_d)
        self.topic_d = ""


def stream_reuters_documents(data_path=None):
    """Iterate over documents of the Reuters dataset.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.
    """
    parser = ReutersParser()
    for filename in glob(os.path.join(data_path, "*.sgm")):
        for doc in parser.parse(open(filename, 'rb')):
            yield doc


def get_topics_from_reuters_documents(data_path):
    fp = os.path.join(data_path, 'all-topics-strings.lc.txt')
    with open(fp) as f:
        for line in f.readlines():
            yield line.rstrip()


class ReutersStreamer:

    def __init__(self, data_path: str=None):
        """

        :param data_path: path to reuters 21578 dataset. If path is not exists then
        """
        self._data_path = data_path
        self._topics = None

    def extract_topics(self) -> List[str]:
        return list(get_topics_from_reuters_documents(self.data_path))

    @property
    def topics(self):
        if self._topics is None:
            self._topics = self.extract_topics()
        return self._topics

    @property
    def data_path(self):
        """
        If `data_path` was passed when class is initialized then will be returned
        If no – dataset will be downloaded into temp directory and
        :return:
        """

        if self._data_path and os.path.exists(self._data_path):
            return self._data_path

        if not self._data_path:
            LOGGER.info('data_path is not specified. Dataset will be downloaded into created temp directory')
            tmp_dir = tempfile.mkdtemp()
            self._data_path = tmp_dir

        elif not os.path.exists(self._data_path):
            LOGGER.info('data_path is specified but no file/directory exists. '
                        'Folder will be created and dataset will be downloaded into it')
            os.mkdir(self._data_path)

        print("Downloading dataset (once and for all) into %s" % self._data_path)

        def progress(blocknum, bs, size):
            total_sz_mb = '%.2f MB' % (size / 1e6)
            current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
            sys.stdout.write(
                '\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb))

        archive_path = os.path.join(self._data_path, ARCHIVE_FILENAME)
        urlretrieve(DOWNLOAD_URL, filename=archive_path,
                    reporthook=progress)
        print("untarring Reuters dataset...")
        tarfile.open(archive_path, 'r:gz').extractall(self._data_path)
        print("done.")

        return self._data_path

    def stream_reuters_documents_with_topics(self) -> Tuple[str, List[str]]:
        """
        Stream only reuters documents where topics are not empty.
        Return tuple: body, topics
        """
        for doc in stream_reuters_documents(self.data_path):
            body, topics = doc['body'], doc['topics']
            if topics:
                yield body, topics





