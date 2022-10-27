import pandas as pd
import pdfkit
import pandas as pd
import time
import os

def Csv_pdf(file_path):
	if file_path.endswith('csv') or file_path.endswith('CSV'):
		timestr = time.strftime("%Y%m%d_%H%M%S")
		path = os.getcwd()+"/"+timestr+'.json'
		df = pd.read_csv (file_path)
		print("The dataframe is:")
		print(df)
		html_string = df.to_html()
		pdfkit.from_string(html_string, path)
		print("PDF file saved.")
		return path
	else:
		print("File is not CSV ")
		return 0