from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser, PDFDocument
import string
import os
from collections import Counter
from functools import reduce
import pandas as pd


def parse(path):
    fp = open(path, 'rb')  # 以二进制读模式打开
    # 用文件对象来创建一个pdf文档分析器
    praser = PDFParser(fp)
    # 创建一个PDF文档
    doc = PDFDocument()
    # 连接分析器 与文档对象
    praser.set_document(doc)
    doc.set_parser(praser)

    # 提供初始化密码
    # 如果没有密码 就创建一个空的字符串
    doc.initialize()

    # 检测文档是否提供txt转换，不提供就忽略
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        # 创建PDf 资源管理器 来管理共享资源
        rsrcmgr = PDFResourceManager()
        # 创建一个PDF设备对象
        laparams = LAParams(all_texts=True)
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        # 创建一个PDF解释器对象
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        # 循环遍历列表，每次处理一个page的内容
        data = []
        for page in doc.get_pages():  # doc.get_pages() 获取page列表
            interpreter.process_page(page)
            # 接受该页面的LTPage对象
            layout = device.get_result()
            # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等 想要获取文本就获得对象的text属性，
            for x in layout:
                if isinstance(x, LTTextBoxHorizontal):
                    results = x.get_text()
                    data.append(results)
        return data


if __name__ == '__main__':
    def getwords(res):
        chars = string.ascii_letters + ' '

        def _filter(x):
            t = x.replace('\t', '').replace('\n', '')
            t = ''.join(filter(lambda x: x in chars, t))
            return t

        res = list(map(_filter, res))
        tmp = []
        for i in res:
            for l in i.split(' '):
                tmp.append(l)
        return Counter(tmp)


    def dict_add(a, b):
        for k, v in b.items():
            if k in a:
                a[k] += v
            else:
                a[k] = v
        return a

    def remove_plural(dct):
        lst = dct.keys()
        for k in lst:
            if len(k) > 1:
                if k.endswith('s'):
                    if k[:-1] in dct and not k.endswith('es'):
                        dct[k[:-1]] += dct[k]
                        dct[k] = 0
                if k.endswith('es'):
                    if k[:-2] in dct:
                        dct[k[:-2]] += dct[k]
                        dct[k] = 0
        return dct

    res_lst = []
    for file in os.listdir('/Users/chamiel/Desktop/雅思阅读pdf/'):
        try:
            res_lst.append(getwords(parse('/Users/chamiel/Desktop/雅思阅读pdf/' + file)))
            print(file, 'done')
        except:
            print(file, 'error')

    res = reduce(dict_add, res_lst)
    res = remove_plural(res)
    res = pd.DataFrame(res, index=[0]).T.reset_index()
    res.columns = ['word', 'count']
    res.sort_values(by='count', ascending=False)
    pd.DataFrame(res).to_csv('/Users/chamiel/Desktop/雅思阅读pdf/res.csv')