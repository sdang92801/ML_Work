import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\Employee\Downloads\athleteEventsNoPersonal.csv')
print(df.groupby(['ID','Year'])['Height'].mean())