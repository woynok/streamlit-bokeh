# [WIP] streamlit-bokeh demo

## What is this?

This is a demo of how to use [streamlit](https://streamlit.io/) and [bokeh](https://docs.bokeh.org/en/2.4.3/index.html) together.

In particular, we focus on creating interactive graphs with CustomJS callbacks.

## TODO

- [x] implement TextData class which receives a text data and perform heavy processing and holds the result
- [x] implement DocumentMap class that performs dimensionality reduction and clustering
- [ ] implement WordStatistic class that performs frequency counting, co-occurrence counting and tf-idf calculation
- [ ] implement bokeh plot for DocumentMap
- [ ] implement bokeh plot for WordStatistic
- [ ] implement BackgroundJob class that runs heavy processing (like TextData method) in background.
