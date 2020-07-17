from PyPDF2 import PdfFileReader, PdfFileWriter
import os


def pdf_splitter(file,outfile):
    pdf = PdfFileReader(file)
    page_num = pdf.getNumPages()
    pdf_writer = PdfFileWriter()
    for i in range(10, page_num):
        pdf_writer.addPage(pdf.getPage(i))
    with open(outfile, 'wb') as out:
        pdf_writer.write(out)


if __name__ == '__main__':
    pdf_splitter('/Users/chamiel/Desktop/雅思阅读pdf/阅读34.pdf','/Users/chamiel/Desktop/雅思阅读pdf/34.pdf')
    pdf_splitter('/Users/chamiel/Desktop/雅思阅读pdf/雅思阅读真题13.pdf','/Users/chamiel/Desktop/雅思阅读pdf/13.pdf')
    pdf_splitter('/Users/chamiel/Desktop/雅思阅读pdf/阅读预测真题14.pdf','/Users/chamiel/Desktop/雅思阅读pdf/14.pdf')
    pdf_splitter('/Users/chamiel/Desktop/雅思阅读pdf/雅思阅读真题17.pdf','/Users/chamiel/Desktop/雅思阅读pdf/17.pdf')

