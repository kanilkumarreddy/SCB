import pandas as pd
import time
import os

def json_text(file_path):
	if file_path.endswith('JSON') or file_path.endswith('json'):
		timestr = time.strftime("%Y%m%d_%H%M%S")
		path =os.getcwd()+"/"+timestr+'.txt'
		df = pd.read_json(file_path)
		df.to_csv( path)
		print(path)
		return path
	else:
		print("File is not JSON ")
		return 0