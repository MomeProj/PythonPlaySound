from matplotlib import font_manager

fonts_set = {f.name for f in font_manager.fontManager.ttflist}
fonts_list = list(fonts_set)
fonts_list.sort(key=str.lower)
for font in fonts_list:
    print(font)
