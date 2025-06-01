import wikipediaapi

wiki = wikipediaapi.Wikipedia('zh')  # 中文版本
page = wiki.page("柯基犬")
print(page.text[:500])  # 打印前500字
