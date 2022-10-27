import pandas as pd
import time
import os
from fpdf import FPDF


def Text_PDF(sentence):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    path = os.getcwd()+"/"+timestr+'.pdf'

    if sentence.endswith('.txt'):
        pdf = FPDF()  
  
        # Add a page
        pdf.add_page()
          
        # set style and size of font
        # that you want in the pdf
        pdf.set_font('Courier','B',16)

         
        # open the text file in read mode
        f = open(sentence, "r")
         
        # insert the texts in pdf
        for x in f:
            pdf.cell(40, 10, txt = x, ln = 1, align = 'C')
          
        # save the pdf with name .pdf
        pdf.output(path)  
        return path

        pdf=FPDF()


    else:
        pdf = FPDF()  
          
        pdf.add_page()
          
        pdf.set_font('Courier','B',16)
         
        f = sentence.split("\n")

        for x in f:
            pdf.cell(40,10, txt = x, ln = 1, align = 'C')
          
        pdf.output(path)  

        return path