# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import string


class English(object):
    chars_uppercase = string.ascii_uppercase
    chars_lowercase = string.ascii_lowercase
    umlauts_uppercase = ''
    umlauts_lowercase = ''
    digits = string.digits
    punctuation = string.punctuation
    newline = '\n'

    def __init__(self, lowercase):
        self.lowercase = lowercase

    def get_alphabet(self):
        components = [self.chars_lowercase, self.umlauts_lowercase,
                      self.digits, self.punctuation, self.newline]
        if not self.lowercase:
            components.extend([self.chars_uppercase, self.umlauts_uppercase])
        return ''.join(sorted(set(''.join(components))))


class German(English):
    umlauts_uppercase = 'ÄÖÜ'
    umlauts_lowercase = 'äöüß'
