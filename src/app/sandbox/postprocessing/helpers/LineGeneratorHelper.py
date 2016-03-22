import numpy as np

class LineGeneratorHelper:
    def __init__(self):
        return

    @staticmethod
    def get_line(start, end):
        """
        Bresenham's Line Algorithm

        :param start: Coordinate (x0, y0)
        :param end: Coordinate (x1, y1)
        :return: Tuples of Coordinates in which the line will pass in a square grid
        """
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        print 'angle of line', np.rad2deg(np.arctan(dy / dx))
        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)
        # print '->', dx, dy, is_steep

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = int(dx / 2.0)
        y_step = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        x1 = int(x1)
        x2 = int(x2)
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()

        return points
