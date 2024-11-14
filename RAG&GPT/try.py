# from fpdf import FPDF
# pdf = FPDF()
# pdf.add_font('simhei','','SIMHEI.TTF',True)
# pdf.set_font("simhei", size=12)
# pdf.add_page()
# pdf.cell(0, 6, "好好学习", 1, ln=0, align="L")
# pdf.cell(0, 6, "two", 1, ln=1, align="R")
# pdf.cell(0, 6, "three", 1, ln=0, align="L")
# pdf.cell(0, 6, "four", ln=1, align="R")
# pdf.output("simple_demo.pdf")

# import matplotlib.font_manager
#
# fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
#
# for font in fonts:
#     print(font)

import requests

proxies = {
    'http': 'http://127.0.0.1:11000',  # HTTP 代理
    'https': 'http://127.0.0.1:11000', # HTTPS 代理
}

response = requests.get('https://huggingface.co', proxies=proxies)
print(response.status_code)

