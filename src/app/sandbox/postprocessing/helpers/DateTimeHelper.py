import ephem


class DateTimeHelper:
    def __init__(self):
        return

    @staticmethod
    def juldate2ephem(date):
        """Convert Julian date to ephem date, measured from noon, Dec. 31, 1899."""
        return ephem.date(date - 2415020.)

    @staticmethod
    def ephem2juldate(date):
        """Convert ephem date (measured from noon, Dec. 31, 1899) to Julian date."""
        return float(date + 2415020.)
