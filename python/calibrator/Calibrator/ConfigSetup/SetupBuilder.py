import numpy as np


class Builder:

    def __init__(self):

        self.buildvers = None

    class sky:

        def __init__(self):

            self.Dir = None             

    class observation:

        def __init__(self):

            self.Chan = None
            self.PCRA = None
            self.PCDec = None
            self.TSteps = None
            self.TStart = None
            self.Length = None
            self.StartFreq = None
            self.IncreFreq = None

    class telescope:

        def __init__(self):

            self.Dir = None
            self.Long = None
            self.Lati = None
            self.PolM = None

    class interferometer:

        def __init__(self):

            self.MS_Dir = None
            self.VisF_Dir = None
            self.Bandwith = None
            self.TAverage = None

    class beam_pattern:

        def __init__(self):

            self.Dir = None
            self.Img_FOV = None
            self.Img_Size = None
            self.CoordFrame = None

    class image:

        def __init__(self):

            self.Dir = None
            self.Img_FOV = None
            self.Img_Size = None
            self.Img_Type = None

