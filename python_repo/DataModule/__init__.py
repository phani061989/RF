from .version import __version__
from .base import data_module_base
from .data_complex import data_complex
from .data_IQ import data_IQ
from .functions import *
from .fit_functions import *
from .data_grid import data_grid

# Init plotting
try:
    from . import plot_style
    from bokeh.resources import INLINE
    from bokeh.io import output_notebook
    output_notebook(INLINE, hide_banner=True)

    # Set default plot style
    plot_style.set()
except:
    pass

# Print version number
print('DataModule v'+__version__)

