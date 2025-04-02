from .mpl_geometry import MplDrawable

class DrawingProperty:
    def __init__(self, fillcolor=None, edgecolor=None, alpha=None, linewidth=None, linestyle=None):
        
        self.edgecolor = edgecolor
        self.fillcolor = fillcolor
        self.alpha = alpha
        self.linewidth = linewidth
        self.linestyle = linestyle


    def __or__(self, other):
        if isinstance(other, MplDrawable):
            other.edgecolor = self.edgecolor if self.edgecolor is not None else other.edgecolor
            other.fillcolor = self.fillcolor if self.fillcolor is not None else other.fillcolor
            other.alpha = self.alpha if self.alpha is not None else other.alpha
            other.linewidth = self.linewidth if self.linewidth is not None else other.linewidth
            other.linestyle = self.linestyle if self.linestyle is not None else other.linestyle

            return other
        
        elif isinstance(other, DrawingProperty):
            return DrawingProperty(
                fillcolor=other.fillcolor if self.fillcolor is None else self.fillcolor,
                edgecolor=other.edgecolor if self.edgecolor is None else self.edgecolor,
                alpha=other.alpha if self.alpha is None else self.alpha,
                linewidth=other.linewidth if self.linewidth is None else self.linewidth,
                linestyle=other.linestyle if self.linestyle is None else self.linestyle
            )
        else:   
            raise TypeError("Unsupported type for | operator")

    def __ror__(self, other):
        if isinstance(other, MplDrawable):
            return self | other
        elif isinstance(other, DrawingProperty):
            return other | self
        else:
            raise TypeError("Unsupported type for | operator")

    def __repr__(self):
        return f"DrawingProperty(color={self.color}, alpha={self.alpha}, linewidth={self.linewidth})"