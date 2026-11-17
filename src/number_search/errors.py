"""Expected errors exposed by the library and handled by the CLI."""


class NumberSearchError(Exception):
    pass


class ImageError(NumberSearchError):
    pass


class ModelError(NumberSearchError):
    pass


class DatasetError(NumberSearchError):
    pass
