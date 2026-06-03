---
title: Blog
---

<h1 class="blog-page-title">Blog</h1>

<ul class="blog-list">
{% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <time datetime="{{ post.date | date: '%Y-%m-%d' }}">{{ post.date | date: "%B %-d, %Y" }}</time>
  </li>
{% endfor %}
</ul>
