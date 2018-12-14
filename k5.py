# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/PUBG_Player_Statistics.csv')
df.head()

df.info()

df.shape

df.describe()

df['solo_KillDeathRatio'].value_counts()
# From this we can see solo kd ratio on avg is high for 1.0


df['solo_RoundsPlayed'].value_counts()
