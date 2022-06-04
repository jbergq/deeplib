import re


RE_ANSI = re.compile(r"\x1b\[[;\d]*[A-Za-z]")


try:
    # TODO consider using wcswidth third-party package for 0-width characters
    from unicodedata import east_asian_width
except ImportError:
    _text_width = len
else:

    def _text_width(s):
        return sum(2 if east_asian_width(ch) in "FW" else 1 for ch in _unicode(s))


def disp_len(data):
    """
    Returns the real on-screen length of a string which may contain
    ANSI control codes and wide chars.
    """
    return _text_width(RE_ANSI.sub("", data))
