from lxml import etree

from .base import BaseExtractor


class TextExtractor(BaseExtractor):
    def feature_representation(self, elem: etree.Element) -> str:
        return self.text_representation(elem)
