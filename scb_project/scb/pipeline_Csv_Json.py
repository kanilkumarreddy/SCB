import pandas as pd
import time
import os

def Csv_json(file_path):
	if file_path.endswith('csv') or file_path.endswith('CSV'):
		timestr = time.strftime("%Y%m%d_%H%M%S")
		path = os.getcwd()+"/"+timestr+'.json'
		df = pd.read_csv (file_path)
		df.to_json ( path)
		print(path)
		return path
	else:
		print("File is not CSV ")
		return 0