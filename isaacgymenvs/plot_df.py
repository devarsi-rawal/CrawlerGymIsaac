from plotter import Plotter
import pandas as pd

plotter = Plotter(1, {})

df = pd.read_csv('runs/crawler-ideal/data-cross/csv/crawler_heading-0.0_swivel-0.0.csv')

logger_vars = {
        "df": df,
        "size": 21, 
        "experiment_dir": "runs/crawler-ideal/"
        }

plotter.dump_states(logger_vars)

# plotter.plot_eval()
plotter._plot_boxplot()
