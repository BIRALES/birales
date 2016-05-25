import ephem


class DateTimeHelper:
    """
    The DateTimeHelper class can be used to convert dates in different formats
    """

    def __init__(self):
        return

    @staticmethod
    def juldate2ephem(date):
        """
        Convert Julian date to ephem date, measured from noon, Dec. 31, 1899.
        :param date: Julian Date
        :return: Date in ephem format
        """
        return ephem.date(date - 2415020.)

    @staticmethod
    def ephem2juldate(date):
        """
        Convert ephem date (measured from noon, Dec. 31, 1899) to Julian date.
        :param date: Ephem Date
        :return: Date in Julian Format
        """
        return float(date + 2415020.)
