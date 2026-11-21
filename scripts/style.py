import matplotlib as mpl
BLUE,ORANGE,AQUA,YELLOW,MAGENTA,GREEN,VIOLET,RED="#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6","#e34948"
INK,INK2,MUTED,GRID,SURF="#0b0b0b","#52514e","#8a8984","#e6e5e1","#fcfcfb"
SEQ=["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]
def setup():
    mpl.rcParams.update({"font.family":"Inter","font.size":10,"axes.edgecolor":GRID,"axes.labelcolor":INK2,
      "xtick.color":INK2,"ytick.color":INK2,"axes.titlesize":12,"axes.titleweight":"bold","axes.titlecolor":INK,
      "axes.titlelocation":"left","axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True,
      "axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white",
      "savefig.dpi":200,"savefig.bbox":"tight","legend.frameon":False})
