# from matplotlib import pyplot as plt
# import matplotlib.cm as cm
# from matplotlib.font_manager import FontProperties
# from sklearn.manifold import TSNE
# from wordcloud import WordCloud
# import random
# from functools import partial
# plt.style.use('seaborn-whitegrid')

# def topic_cloud(word_freqs, hue_index, size=300):  

#   def circle(size=300):
#     d = size
#     r = size // 2
#     m = r * 0.86
#     x, y = np.ogrid[:d, :d]
#     mask = (x - r) ** 2 + (y - r) ** 2 > m ** 2
#     return 255 * mask.astype(int)

#   def random_color_func(word, font_size, position, orientation, 
#                         random_state=None, num_topics=9, hue_index=0, **kwargs):
#     fluc = 30
#     hue = 360 // num_topics * (hue_index) + fluc
#     return "hsl(%d, 100%%, %d%%)" % (hue + random.randint(-fluc, fluc), random.randint(30, 40))

#   fpath = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'
#   fp = FontProperties(fname=fpath, size=16)
#   wordcloud = WordCloud(background_color="white",font_path=fpath, mask=circle(size))
#   wc = wordcloud.generate_from_frequencies(word_freqs[hue_index])
#   color_func = partial(random_color_func, num_topics=len(word_freqs)  ,hue_index=hue_index)
#   return wc.recolor(color_func=color_func)

# def draw_lda_cloud(components, top_k_words, aspect=(10,8), dpi=180, cloud_size_pct=0.2):

#   def draw_topic_cloud(x, y, top_k_words, index, size=100):
#     tc = topic_cloud(top_k_words, hue_index=index, size=300)
#     r = size//2
#     plt.imshow(tc, extent=(x - r, x + r, y -r, y + r))

#   tsne = TSNE(n_components=2)
#   X_2d = tsne.fit_transform(components)

#   xmin = X_2d[:,0].min()
#   xmax = X_2d[:,0].max()
#   ymin = X_2d[:,1].min()
#   ymax = X_2d[:,1].max()
#   xmargine  = (xmax - xmin) * 0.3
#   ymargine = (ymax - ymin) * 0.3
#   cloud_size = ((xmax - xmin) + (ymax - ymin)) / 2 * cloud_size_pct

#   fig, ax = plt.subplots(figsize=aspect, dpi=dpi)

#   for i in range(len(X_2d)):
#     #ax.scatter(X_2d[i][0], X_2d[i][1], s=4, label=i) 
#     draw_topic_cloud(X_2d[i][0], X_2d[i][1], top_k_words, i, size=cloud_size)

#   plt.xlim((int(xmin - xmargine), int(xmax + xmargine)))
#   plt.ylim((int(ymin - ymargine), int(ymax + ymargine)))
#   #plt.axis("off")
#   plt.tick_params(grid_alpha=0.6, grid_linewidth=0.5)
#   plt.show()

#  draw_lda_cloud(lda.components_, top_k_words, cloud_size_pct=0.4)